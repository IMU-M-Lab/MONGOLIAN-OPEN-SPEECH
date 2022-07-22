# -*- encoding: utf-8 -*-
"""
@File       :   gen_SE_data_list.py
@Created by :   lx
@Create Time:   2021/11/15 11:29
@Description:   gen_SE_data_list

data_format:
utt:BAC009S0724W0121_-5_est
feat:/data/lx/data/aishell/processed/test_enhanced/est/BAC009S0905W0183_destroyerops_-5_est.wav
feat_shape:4.2039375
text:投金或在经济增长速度放缓形势下加速外流
token:投 金 或 在 经 济 增 长 速 度 放 缓 形 势 下 加 速 外 流
tokenid:1156 1104 1121 1393 722 85 30 103 433 139 338 1789
token_shape:12,4233
snr:
"""
import codecs
import glob
import os


def mv_file(test_dir, file_type):
    cmd = 'cd {}'.format(test_dir)
    cmd += ' && mkdir {}'.format(file_type)
    cmd += ' && mv *_{}.wav {}/'.format(file_type, file_type)
    os.system(cmd)


def gen_list(test_dir, es_type):
    file_list = glob.glob('{}/{}/*.wav'.format(test_dir, es_type))
    file_dict = {}

    for t in file_list:
        file_name = t.rsplit('/', 1)[1].split('_')[0]
        file_dict[file_name] = t
    format_file = '/data/lx/data/aishell/processed/raw_wav/test/format.data'
    out_file = '/data/lx/data/aishell/processed/test_enhanced/{}_format.data'.format(es_type)
    save_to_file(format_file, out_file, file_dict)


def gen_list_snr(test_dir, es_type, snr):
    file_list = glob.glob('{}/{}/*.wav'.format(test_dir, es_type))
    # snr_file_dict = {'-5': {}, '0': {}, '10': {}, '15': {}}
    snr_file_dict = {s: {} for s in snr}

    for t in file_list:
        file_name = t.rsplit('/', 1)[1].split('_')[0]
        snr_ = t.split('/', 1)[1].split('_')[3]
        snr_file_dict[snr_][file_name] = t

    format_file = '/data/lx/data/aishell/processed/raw_wav/test/format.data'
    num = 0
    for s in snr:
        out_file = '/data/lx/data/aishell/processed/test_enhanced/{}_{}_format.data'.format(es_type, s)
        save_to_file(format_file, out_file, snr_file_dict[s])
        num += len(snr_file_dict[s])
    print(num)
    print(len(file_list))


def save_to_file(format_file, out_file, file_dict: dict):
    with codecs.open(format_file, 'r', encoding='utf-8') as fin:
        with codecs.open(out_file, 'w', encoding='utf-8') as fout:
            for line in fin:
                arr = line.strip().split('\t')
                utt = arr[0].split(':')[1]
                try:
                    arr[0] = 'utt:' + file_dict[utt].rsplit('/', 1)[1]
                    arr[1] = 'feat:' + file_dict[utt]
                    arr.append('snr:' + file_dict[utt].rsplit('_', 2)[1])
                    arr_out = "\t".join(arr)
                    fout.write(arr_out + '\n')
                except KeyError:
                    print(utt)


def main():
    snr = ['-5', '0', '5', '10', '15']
    test_dir = '/data/lx/data/aishell/processed/test_enhanced'
    types = [
        'est',
        'mix',
    ]
    for tp in types:
        mv_file(test_dir, tp)
        # gen_list(test_dir, tp)
        gen_list_snr(test_dir, tp, snr)


if __name__ == '__main__':
    main()
