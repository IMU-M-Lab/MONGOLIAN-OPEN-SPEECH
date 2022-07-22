# -*- encoding: utf-8 -*-
"""
@File       :   lm_dataset.py
@Created by :   lx
@Create Time:   2021/11/18 21:08
@Description:   lm_dataset
"""
from torch.utils.data import Dataset, DataLoader
import codecs
import csv


class CollateFunc(object):
    # def __init__(self):
    #     pass

    def __call__(self, batch):
        print(batch)


# class LMDataset(Dataset):
#     def __init__(self, file_path):
#         self.data_dict = {''}
#         with codecs.open(file_path, 'w', encoding='utf-8') as fin:
#             reader = csv.reader(fin)
#             for line in reader:
#
#
#     def __len__(self):
#         return len(self.minibatch)
#
#     def __getitem__(self, idx):
#         return self.minibatch[idx]


def data_prepare():
    sets = [
        'train',
        'dev',
        'test',
    ]
    for s in sets:
        in_file = '/data/lx/data/aishell/processed/raw_wav/{}/format.data'.format(s)
        out_file = '/data/lx/data/aishell/processed/raw_wav/{}/albert_format.data'.format(s)
        with codecs.open(in_file, encoding='utf-8') as fin:
            with codecs.open(out_file, 'w', encoding='utf-8') as fout:
                writer = csv.writer(fout)
                writer.writerow(['id', 'file', 'labels'])
                for line in fin:
                    arr = line.strip().split('\t')
                    wav_id = arr[0].split(':')[1]
                    file_path = arr[1].split(':')[1]
                    token = arr[4].split(':')[1]
                    writer.writerow((wav_id, file_path, token))


# if __name__ == '__main__':
#     data_prepare()
