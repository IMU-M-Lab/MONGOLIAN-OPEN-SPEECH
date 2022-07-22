#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File       :   convert_model.py
@Created by :   lx
@Create Time:   2021/9/28 19:52
@Description:   加载模型的一部分；比对模型
"""

import os
import sys
sys.path.append(sys.path[0].rsplit('/', 1)[0])
from utils.checkpoint import save_model, load_model
from hyperpyyaml import load_hyperpyyaml
from model.base_model import init_asr_model
import torch


def load_part_model(model, path):
    pretrained_dict = torch.load(path, map_location='cpu')
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)


def convert_model(target_params_file, src_model_path):
    with open(target_params_file, 'r') as fin:
        configs = load_hyperpyyaml(fin)  # overrides 的改动不会保存到yaml里

    save_model_path = os.path.join(configs['output_folder'], 'latest.pt')
    if not os.path.exists(configs['output_folder']):
        os.makedirs(configs['output_folder'])
    input_dim = configs['collate_conf']['feature_extraction_conf']['mel_bins']
    configs['model_init_conf']['input_dim'] = input_dim
    configs['model_init_conf']['output_dim'] = 4233

    model = init_asr_model(configs['model_init_conf'])

    load_part_model(model, src_model_path)
    save_model(model, save_model_path)
    print('convert model to ' + save_model_path)


def compare_model(model1_params_file, model2_params_file):
    """
    model1 should include model2
    """

    with open(model1_params_file, 'r') as fin:
        model1_configs = load_hyperpyyaml(fin)  # overrides 的改动不会保存到yaml里
    # input_dim = model1_configs['collate_conf']['feature_extraction_conf']['mel_bins']
    # model1_configs['model_init_conf']['input_dim'] = input_dim
    # model1_configs['model_init_conf']['output_dim'] = 4233
    # model = init_asr_model(model1_configs['model_init_conf'])

    with open(model2_params_file, 'r') as fin:
        model2_configs = load_hyperpyyaml(fin)  # overrides 的改动不会保存到yaml里
    # input_dim = model2_configs['collate_conf']['feature_extraction_conf']['mel_bins']
    # model2_configs['model_init_conf']['input_dim'] = input_dim
    # model2_configs['model_init_conf']['output_dim'] = 4233
    # model = init_asr_model(model2_configs['model_init_conf'])

    model1_path = model1_configs['checkpoint']
    model2_path = model2_configs['checkpoint']
    model1_dict = torch.load(model1_path, map_location='cpu')
    model2_dict = torch.load(model2_path, map_location='cpu')
    # for param in model_2.encoder.parameters():
    #     param.requires_grad = False
    # model_2.load_state_dict(model2_dict)
    # for param in model_2.encoder.parameters():
    #     print(param.requires_grad)
    for k in model1_dict.keys():
        if k not in model2_dict:
            print('key:{} not in model2')
        elif not (model1_dict[k] == model2_dict[k]).min().item():
            print('key:{} not equal'.format(k))



