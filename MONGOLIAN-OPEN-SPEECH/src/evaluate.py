# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Xiaoyu Chen, Di Wu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import argparse
import copy
import logging
import os.path
import sys
import codecs
import csv
import re
import math

import torch
from hyperpyyaml import load_hyperpyyaml
from torch.utils.data import DataLoader

from dataset.dataset import Dataset
from model.base_model import init_asr_model
from utils.checkpoint import load_model
from utils.avg_model import avg_model
from utils.file_utils import read_symbol_table, read_non_lang_symbols
from metric.metric import ErrorRateMetric
from tqdm import tqdm
from metric.ml_and_metric import ml_process,compute_cwer

def parse_arguments():
    parser = argparse.ArgumentParser(description='recognize with your model')
    parser.add_argument('--param_file', required=True, help='param_file file')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--checkpoint',
                        help='checkpoint model')
    parser.add_argument('--dict', help='dict file')
    parser.add_argument("--non_lang_syms",
                        help="non-linguistic symbol file. One symbol per line.")
    parser.add_argument('--avg.avg_model',
                        dest='avg_model',
                        action='store_true',
                        help='use avg_model')
    parser.add_argument('--avg.val_best',
                        dest='val_best',
                        action='store_true',
                        help='use best val score to avg.')
    parser.add_argument('--avg.num',
                        default=5,
                        type=int,
                        dest='num',
                        help='nums for averaged model')
    parser.add_argument('--avg.min_epoch',
                        default=0,
                        type=int,
                        dest='min_epoch',
                        help='min epoch used for averaging model')
    parser.add_argument('--avg.max_epoch',
                        default=65536,  # Big enough
                        type=int,
                        dest='max_epoch',
                        help='max epoch used for averaging model')
    parser.add_argument('--beam_size',
                        type=int,
                        default=10,
                        help='beam size for search')
    parser.add_argument('--penalty',
                        type=float,
                        default=0.0,
                        help='length penalty')
    parser.add_argument('--result_file', help='asr result file')
    parser.add_argument('--batch_size',
                        type=int,
                        default=1,
                        help='asr result file')
    parser.add_argument('--mode',
                        choices=[
                            'attention', 'ctc_greedy_search',
                            'ctc_prefix_beam_search', 'attention_rescoring'
                        ],
                        default='attention_rescoring',
                        help='decoding mode')
    parser.add_argument('--ctc_weight',
                        type=float,
                        default=0.0,
                        help='ctc weight for attention rescoring decode mode')
    parser.add_argument('--decoding_chunk_size',
                        type=int,
                        default=-1,
                        help='''decoding chunk size,
                                    <0: for decoding, use full chunk.
                                    >0: for decoding, use fixed chunk size as set.
                                    0: used for training, it's prohibited here''')
    parser.add_argument('--num_decoding_left_chunks',
                        type=int,
                        default=-1,
                        help='number of left chunks for decoding')
    parser.add_argument('--simulate_streaming',
                        action='store_true',
                        help='simulate streaming inference')
    parser.add_argument('--test_data',
                        help='test dateset')
    parser.add_argument('--bpe_model',
                        default=None,
                        type=str,
                        help='bpe model for english part')
    parser.add_argument('--out_name_suffix',
                        default='',
                        help='result file suffix')
    parser.add_argument('--output_folder',
                        help='output folder')
    parser.add_argument('--reverse_weight',
                        type=float,
                        default=0.0,
                        help='''right to left weight for attention rescoring
                                    decode mode''')
    args = parser.parse_args()
    param_file = args.param_file
    print(args)
    override = {}
    if args.checkpoint is not None:
        override['checkpoint'] = args.checkpoint
    if args.test_data is not None:
        override['test_data'] = args.test_data
    if args.output_folder is not None:
        override['output_folder'] = args.output_folder
    if args.batch_size is not None:
        override['dataset_conf'] = {'batch_conf': {'batch_size': args.batch_size}}
    return args, param_file, override


def token2text(token, char_dict, eos):
    return ''.join([char_dict[t] for t in token if t != eos])
    # text = ''
    # for w in token:
    #
    #     if w == eos:
    #         break
    #     text += char_dict[w]
    # return text


def main():
    args, params_file, override = parse_arguments()

    with open(params_file, 'r') as fin:
        configs = load_hyperpyyaml(fin, override)
    if args.avg_model:
        src_path = configs.get('output_folder')
        dst_model = os.path.join(configs.get('output_folder'), 'avg_model.pt')
        avg_model(dst_model=dst_model, src_path=src_path, val_best=args.val_best,
                  num=args.num, min_epoch=args.min_epoch, max_epoch=args.max_epoch)

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')

    if args.mode in ['ctc_prefix_beam_search', 'attention_rescoring'
                     ] and args.batch_size > 1:
        logging.fatal(
            'decoding mode {} must be running with batch_size == 1'.format(
                args.mode))
        sys.exit(1)

    symbol_table = read_symbol_table(configs['dict_dir'])
    test_conf = copy.deepcopy(configs['dataset_conf'])

    test_conf['filter_conf']['max_length'] = 102400
    test_conf['filter_conf']['min_length'] = 0
    test_conf['filter_conf']['token_max_length'] = 102400
    test_conf['filter_conf']['token_min_length'] = 0
    test_conf['filter_conf']['max_output_input_ratio'] = 102400
    test_conf['filter_conf']['min_output_input_ratio'] = 0
    test_conf['speed_perturb'] = False
    test_conf['spec_aug'] = False
    test_conf['shuffle'] = False
    test_conf['sort'] = False
    test_conf['fbank_conf']['dither'] = 0.0
    test_conf['batch_conf']['batch_type'] = "static"
    non_lang_syms = read_non_lang_symbols(args.non_lang_syms)

    test_dataset, test_size = Dataset(configs['data_type'],
                                      configs['test_data'],
                                      symbol_table,
                                      test_conf,
                                      configs.get('bpe_model', None),
                                      non_lang_syms,
                                      partition=False)

    test_data_loader = DataLoader(test_dataset, batch_size=None, num_workers=0)

    val_b = '_val_best' if args.val_best else '_val_last'
    result_filename = str(args.mode + val_b + '_' + args.out_name_suffix + '_result.csv')
    result_path = os.path.join(configs['output_folder'], result_filename)

    error_rate_metric = ErrorRateMetric()

    vocab_size = len(symbol_table)
    if configs['data_type'] == 'raw':
        input_dim = configs['collate_conf']['feature_extraction_conf']['mel_bins']
    else:
        input_dim = test_dataset.input_dim

    configs['model_init_conf']['input_dim'] = input_dim
    configs['model_init_conf']['output_dim'] = vocab_size

    # Init asr model from configs
    model = init_asr_model(configs['model_init_conf'])

    # Load dict
    char_dict = {}
    with open(configs['dict_dir'], 'r') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            char_dict[int(arr[1])] = arr[0]
    eos = len(char_dict) - 1

    if args.avg_model:
        model_path = re.sub(r'\w+.pt$', 'avg_model.pt', configs['checkpoint'])
    else:
        model_path = configs['checkpoint']

    batch_size = configs['dataset_conf']['batch_conf']['batch_size']
    batch_num = math.ceil(test_size / batch_size) + 1

    load_model(model, model_path)
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    model = model.to(device)

    model.eval()
    real = []
    est = []
    with torch.no_grad(), open(result_path, 'w', encoding='utf8') as fout:
        writer = csv.writer(fout)
        writer.writerow(['key', 'prediction', 'utt_cer', 'all_cer'])
        pbar = tqdm(test_data_loader, total=batch_num)
        for batch_idx, batch in enumerate(pbar):
            keys = batch['keys']
            speech = batch['speech'].to(device)
            speech_lengths = batch['speech_lengths'].to(device)
            text = batch['text']
            if args.mode == 'attention':
                hyps, _ = model.recognize(
                    speech,
                    speech_lengths,
                    beam_size=args.beam_size,
                    decoding_chunk_size=args.decoding_chunk_size,
                    num_decoding_left_chunks=args.num_decoding_left_chunks,
                    simulate_streaming=args.simulate_streaming)
                hyps = [hyp.tolist() for hyp in hyps]
            elif args.mode == 'ctc_greedy_search':
                hyps, _ = model.ctc_greedy_search(
                    speech,
                    speech_lengths,
                    decoding_chunk_size=args.decoding_chunk_size,
                    num_decoding_left_chunks=args.num_decoding_left_chunks,
                    simulate_streaming=args.simulate_streaming)
                for hyp in hyps:
                    if len(hyp) != 0:
                        hyp = hyp[:-1] if hyp[-1] == eos else hyp
                    else:
                        hyp = [' ']
            # ctc_prefix_beam_search and attention_rescoring only return one
            # result in List[int], change it to List[List[int]] for compatible
            # with other batch decoding mode
            elif args.mode == 'ctc_prefix_beam_search':
                assert (speech.size(0) == 1)
                hyp, _ = model.ctc_prefix_beam_search(
                    speech,
                    speech_lengths,
                    args.beam_size,
                    decoding_chunk_size=args.decoding_chunk_size,
                    num_decoding_left_chunks=args.num_decoding_left_chunks,
                    simulate_streaming=args.simulate_streaming)
                hyps = [hyp]
            elif args.mode == 'attention_rescoring':
                assert (speech.size(0) == 1)
                hyp, _ = model.attention_rescoring(
                    speech,
                    speech_lengths,
                    args.beam_size,
                    decoding_chunk_size=args.decoding_chunk_size,
                    num_decoding_left_chunks=args.num_decoding_left_chunks,
                    ctc_weight=args.ctc_weight,
                    simulate_streaming=args.simulate_streaming,
                    reverse_weight=args.reverse_weight)
                hyps = [hyp]

            for i, key in enumerate(keys):
                est_text = ''.join([char_dict[t] for t in hyps[i] if t != eos])
                tmp = list(text[0])
                real_text = ''.join([char_dict[int(t)] for t in tmp if t != eos])
                # logging.info('{} {}'.format(key, content))
                # fout.write('{} {}\n'.format(key, content))
                l = text.numpy().tolist()[i]
                l = l[:l.index(-1)] if -1 in l else l

                ref = ml_process(real_text)
                rec = ml_process(est_text)
                real.append(ref)
                est.append(rec)

                wer,cer = compute_cwer(ref,rec)
                # one_wer = error_rate_metric.one_utt_cer(l, hyps[i])
                # all_wer = error_rate_metric.all_cer()

                # strtmp = ''.join(" 真实值：{}，预测值：{},wer:{}")
                fout.write("key:{}, 真实值：{}，预测值：{},wer:{}".format(key, real_text, est_text, wer))


                logging.info(
                    "key:{}, 真实值：{}，预测值：{},cer:{},wer:{}".format(key, real_text, est_text, cer, wer))

    all_wer, all_cer = compute_cwer(real, est)
    pbar.set_postfix({'all_wer': all_wer})


if __name__ == '__main__':
    main()
