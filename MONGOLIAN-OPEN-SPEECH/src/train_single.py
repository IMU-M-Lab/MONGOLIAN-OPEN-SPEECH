#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File       :   train_single.py
@Created by :   lx

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/9/17 18:06   lx         0.1         None
"""

from __future__ import print_function

import argparse
import copy
import logging
import os
from hyperpyyaml import load_hyperpyyaml

import torch
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from dataset.dataset import Dataset
from model.base_model import init_asr_model
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.executor import Executor
from utils.scheduler import WarmupLR
from utils.file_utils import read_symbol_table


def parse_arguments():
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument("--param_file",
                        type=str,
                        required=True,
                        help="A yaml-formatted file, config file. ", )
    parser.add_argument('--data_type',
                        default=None,
                        choices=['raw', 'shard'],
                        help='train and cv data type')
    parser.add_argument('--ddp.rank',
                        dest='rank',
                        type=int,
                        help='global rank for distributed training')
    parser.add_argument('--ddp.world_size',
                        dest='world_size',
                        type=int,
                        help='''number of total processes/gpus for
                            distributed training''')
    parser.add_argument('--ddp.dist_backend',
                        choices=['nccl', 'gloo'],
                        dest='dist_backend',
                        help='distributed backend')
    parser.add_argument('--ddp.init_method',
                        default=None,
                        dest='init_method',
                        help='ddp init method')
    parser.add_argument('--use_cpu',
                        action="store_true")
    parser.add_argument('--prefetch',
                        default=None,
                        type=int,
                        help='prefetch number')
    parser.add_argument('--symbol_table',
                        help='model unit symbol table for training')
    # parser.add_argument('--use_amp',
    #                     action='store_true',
    #                     default=False,
    #                     help='Use automatic mixed precision training')
    # parser.add_argument('--num_workers',
    #                     default=0,
    #                     type=int,
    #                     help='num of subprocess workers for reading')
    # parser.add_argument('--pin_memory',
    #                     action='store_true',
    #                     default=False,
    #                     help='Use pinned memory buffers used for reading')
    #
    # parser.add_argument('--cmvn', default=None, help='global cmvn file')

    args = parser.parse_args()
    param_file = args.param_file

    return args, param_file


def main(configs, use_cpu):
    symbol_table = read_symbol_table(configs['dict_dir'])

    train_conf = configs['dataset_conf']
    cv_conf = copy.deepcopy(train_conf)
    cv_conf['speed_perturb'] = False
    cv_conf['spec_aug'] = False

    train_dataset, train_size = Dataset(configs['data_type'], configs['train_data'], symbol_table,
                            train_conf, configs.get('bpe_model', None), partition=True)
    cv_dataset, cv_size = Dataset(configs['data_type'],
                         configs['cv_data'],
                         symbol_table,
                         cv_conf,
                         configs.get('bpe_model', None),
                         partition=False)

    train_data_loader = DataLoader(train_dataset,
                                   batch_size=None,
                                   pin_memory=configs['pin_memory'],
                                   num_workers=configs['num_workers'],
                                   )
    cv_data_loader = DataLoader(cv_dataset,
                                batch_size=None,
                                pin_memory=configs['pin_memory'],
                                num_workers=configs['num_workers'],
                                )





    input_dim = configs['dataset_conf']['fbank_conf']['num_mel_bins']
    vocab_size = len(symbol_table)

    # Save configs to model_dir/train.yaml for inference and export
    model_dir = configs["output_folder"]

    configs['model_init_conf']['input_dim'] = input_dim
    configs['model_init_conf']['output_dim'] = vocab_size
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    saved_config_path = os.path.join(model_dir, 'train.yaml')
    with open(saved_config_path, 'w') as fout:
        data = yaml.dump(configs)
        fout.write(data)

    # Init asr model from configs
    model = init_asr_model(configs['model_init_conf'])
    # print(model)
    num_params = sum(p.numel() for p in model.parameters())
    print('the number of model params: {}'.format(num_params))

    executor = Executor()

    if not use_cpu and not torch.cuda.is_available():
        logging.error('cuda not available')

    # If specify checkpoint, load some info from checkpoint
    if configs['use_ckpt']:
        if os.path.isfile(configs['checkpoint']):
            infos = load_checkpoint(model, configs['checkpoint'], use_cpu)
        else:
            logging.error('ckpt file not exits')
            logging.error('if not use ckpt, please set use_ckpt: False')
            return
    else:
        infos = {}
    start_epoch = int(infos.get('epoch', -1)) + 1
    cv_loss = float(infos.get('cv_loss', 0.0))
    step = int(infos.get('step', -1))
    print('start_epoch:' + str(start_epoch) + ' cv_loss:' + str(cv_loss))

    num_epochs = configs.get('max_epoch', 100)

    writer = None
    os.makedirs(model_dir, exist_ok=True)
    exp_id = os.path.basename(model_dir)
    writer = SummaryWriter(os.path.join(configs['tensorboard'], exp_id))

    use_cuda = not use_cpu and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = model.to(device)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), **configs['optim_conf'])
    scheduler = WarmupLR(optimizer, **configs['scheduler_conf'])
    final_epoch = None
    configs['is_distributed'] = False
    if start_epoch == 0:
        save_model_path = os.path.join(model_dir, 'init.pt')
        save_checkpoint(model, save_model_path)

    # Start training loop
    executor.step = step
    scheduler.set_step(step)
    # used for pytorch amp mixed precision training
    scaler = None
    if configs['use_amp']:
        scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, num_epochs):
        train_dataset.set_epoch(epoch)
        # configs['epoch'] = epoch
        lr = optimizer.param_groups[0]['lr']
        logging.info('Epoch {} TRAIN info lr {}'.format(epoch, lr))
        executor.train(model, optimizer, scheduler, train_data_loader, device,
                       writer, configs, scaler)
        total_loss, num_seen_utts = executor.cv(model, cv_data_loader, device,
                                                configs)
        cv_loss = total_loss / num_seen_utts

        logging.info('Epoch {} CV info cv_loss {}'.format(epoch, cv_loss))
        if epoch % configs['ckpt_interval_epoch'] == 0:
            save_model_path = os.path.join(model_dir, '{}.pt'.format(epoch))
            save_checkpoint(
                model, save_model_path, {
                    'epoch': epoch,
                    'lr': lr,
                    'cv_loss': cv_loss,
                    'step': executor.step
                })
        writer.add_scalars('epoch', {'cv_loss': cv_loss, 'lr': lr}, epoch)
        final_epoch = epoch

    if final_epoch is not None:
        final_model_path = os.path.join(model_dir, 'final.pt')
        os.symlink('{}.pt'.format(final_epoch), final_model_path)
        writer.close()


if __name__ == '__main__':
    args, params_file = parse_arguments()

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    # Set random seed
    # print(args)

    with open(params_file, 'r') as fin:
        configs = load_hyperpyyaml(fin)  # overrides 的改动不会保存到yaml里
    configs['num_workers'] = 0  # 调试时改为0
    if not os.path.exists(configs['cmvn_file']):
        logging.error('cmvn file not exit')
        raise FileNotFoundError
    main(configs, args.use_cpu)
