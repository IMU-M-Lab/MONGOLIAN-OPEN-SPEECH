# Copyright 2019 Mobvoi Inc. All Rights Reserved.
# Author: di.wu@mobvoi.com (DI WU)
import os
import argparse
import glob
import time

import yaml
import numpy as np
import torch
from hyperpyyaml import load_hyperpyyaml


def avg_model(dst_model, src_path, val_best, num, min_epoch, max_epoch):
    checkpoints = []
    val_scores = []
    if val_best:
        info_path = os.path.join(src_path,'info.csv')
        with open(info_path, 'r') as f:
            f.readline()
            for line in f:
                print(line)
                arr = line.split(',')
                epoch = int(arr[0])
                loss = float(arr[1])
                if min_epoch <= epoch <= max_epoch:
                    val_scores += [[epoch, loss]]
        val_scores = np.array(val_scores)
        sort_idx = np.argsort(val_scores[:, -1])
        sorted_val_scores = val_scores[sort_idx][::1]
        print("best val scores = " + str(sorted_val_scores[:num, 1]))
        print("selected epochs = " +
              str(sorted_val_scores[:num, 0].astype(np.int64)))
        path_list = [
            src_path + '/{}.pt'.format(int(epoch))
            for epoch in sorted_val_scores[:num, 0]
        ]
    else:
        path_list = glob.glob('{}/[!avg][!final]*.pt'.format(src_path))
        path_list = sorted(path_list, key=os.path.getmtime)
        path_list = path_list[-num:]
    print(path_list)
    avg = None
    assert num == len(path_list)
    for path in path_list:
        print('Processing {}'.format(path))
        states = torch.load(path, map_location=torch.device('cpu'))
        if avg is None:
            avg = states
        else:
            for k in avg.keys():
                avg[k] += states[k]
    # average
    for k in avg.keys():
        if avg[k] is not None:
            # pytorch 1.6 use true_divide instead of /=
            avg[k] = torch.true_divide(avg[k], num)
    print('Saving to {}'.format(dst_model))
    torch.save(avg, dst_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='average model')
    # parser.add_argument('--dst_model', required=True, help='averaged model')
    # parser.add_argument('--src_path',
    #                     required=True,
    #                     help='src model path for average')
    parser.add_argument('--param_file',
                        help='param file path',
                        required=True)
    parser.add_argument('--val_best',
                        action="store_true",
                        help='averaged model')
    parser.add_argument('--num',
                        default=5,
                        type=int,
                        help='nums for averaged model')
    parser.add_argument('--min_epoch',
                        default=0,
                        type=int,
                        help='min epoch used for averaging model')
    parser.add_argument('--max_epoch',
                        default=65536,  # Big enough
                        type=int,
                        help='max epoch used for averaging model')

    args = parser.parse_args()
    print(args)
    params_file = args.param_file
    with open(params_file, 'r') as fin:
        configs = load_hyperpyyaml(fin)
    src_path = configs.get('output_folder')
    dst_model = os.path.join(configs.get('output_folder'), 'avg_model.pt')
    avg_model(dst_model=dst_model, src_path=src_path, val_best=args.val_best,
              num=args.num, min_epoch=args.min_epoch, max_epoch=args.max_epoch)
