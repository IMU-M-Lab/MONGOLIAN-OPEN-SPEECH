# Copyright 2019 Mobvoi Inc. All Rights Reserved.
# Author: binbinzhang@mobvoi.com (Binbin Zhang)

import logging
import shutil
import os
import re
import csv

import yaml
import torch


def _load_info(path: str):
    info_path = re.sub(r'\w*.pt$', 'info.csv', path)
    which = re.search(r'(\w*).pt$', path).group(1)
    try:
        result = None
        f = open(info_path, 'r')
        reader = csv.DictReader(f)
        if which == 'latest':
            *_, last = reader  # get the last one
            result = last
        else:
            for item in reader:
                if item.get('epoch') == which:
                    result = item
                    break
        assert result is not None
    except IOError:
        result = {'epoch': 0, 'cv_loss': 0.0, 'step': -1}
    return result


def load_model(model: torch.nn.Module, path: str, use_cpu=False):
    if torch.cuda.is_available() and not use_cpu:
        logging.info('Checkpoint: loading from checkpoint %s for GPU' % path)
        checkpoint = torch.load(path)
    else:
        logging.info('Checkpoint: loading from checkpoint %s for CPU' % path)
        checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint)
    print('load model from:' + path)


def load_checkpoint(model: torch.nn.Module, path: str, use_cpu=False) -> dict:
    infos = _load_info(path)
    start_epoch = infos.get('epoch')
    if start_epoch == 0:
        re.sub(r'latest.pt$', 'init.pt', path)
    load_model(model, path, use_cpu)
    return infos


# def save_checkpoint(model: torch.nn.Module, path: str, infos=None):
#     '''
#     Args:
#         infos (dict or None): any info you want to save.
#     '''
#     logging.info('Checkpoint: save to checkpoint %s' % path)
#     if isinstance(model, torch.nn.DataParallel):
#         state_dict = model.module.state_dict()
#     elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
#         state_dict = model.module.state_dict()
#     else:
#         state_dict = model.state_dict()
#     torch.save(state_dict, path)
#     if 'init' in path:
#         return
#     latest_path = re.sub(r'\d+.pt$', 'latest.pt', path)
#     torch.save(state_dict, latest_path)
#     info_path = re.sub(r'\d+.pt$', 'info.csv', path)
#     if infos is None:
#         infos = {}
#     with open(info_path, 'a') as f:
#         writer = csv.writer(f)
#         if f.tell() == 0:
#             writer.writerow(['epoch', 'cv_loss', 'lr', 'step'])
#         writer.writerow([infos.get('epoch', 0), infos.get('cv_loss', 0.),
#                          infos.get('lr', 0.), infos.get('step', 0)])
#         f.close()
def save_model(model: torch.nn.Module, path: str, ):
    """
    save model
    Args:
        :param path: save path
        :param model: model
        :return: state_dict
    """
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(state_dict, path)
    return state_dict


def save_checkpoint(model: torch.nn.Module, path: str, infos=None):
    """
    save model and information
    Args:
        infos (dict or None): any info you want to save.
        :param infos: {epoch,lr,cv_loss,step}
        :param path: save path
        :param model: model
    """
    state_dict = save_model(model, path)
    if 'init' in path:
        return

    latest_path = re.sub(r'\d+.pt$', 'latest.pt', path)
    torch.save(state_dict, latest_path)

    info_path = re.sub(r'\w+.pt$', 'info.csv', path)
    if infos is None:
        infos = {}
    with open(info_path, 'a') as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(['epoch', 'cv_loss', 'lr', 'step'])
        writer.writerow([infos.get('epoch', 0), infos.get('cv_loss', 0.),
                         infos.get('lr', 0.), infos.get('step', 0)])
        f.close()
