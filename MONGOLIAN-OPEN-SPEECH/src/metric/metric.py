#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File       :   metric.py
@Created by :   lx
@Create Time:   2021/9/26 16:17
@Description:   调用编辑距离计算指标
"""
from metric1.edit_distance import op_table, count_ops

import os.path
import csv


class ErrorRateMetric:
    """
    A class for tracking error rates (e.g., WER, PER).
    """

    def __init__(self, save_path=None):
        self.numerator = 0
        self.denominator = 0
        self.num = 0
        self.path = save_path
        if save_path is not None and os.path.isfile(save_path):
            with open(save_path, 'w'):
                print('清空原有数据')


    def one_utt_cer(self, refs, est):
        counter = count_ops(op_table(refs, est))
        numerator = 0
        for key in counter.keys():
            numerator += counter[key]
        # print(counter)
        self.numerator += numerator
        self.denominator += len(refs)
        self.num += 1
        return (numerator / len(refs)) * 100

    def all_cer(self):
        return (self.numerator / self.denominator) * 100

    def save_info(self, info):
        with open(self.path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([info, 100 * (self.numerator / self.denominator)])
            f.close()


if __name__ == '__main__':
    c = ['ᠴᠠᠬᠢᠯᠭᠠᠨ', 'ᠲᠠᠢ', 'ᠪᠠ', 'ᠴᠠᠬᠢᠯᠭᠠᠨ', 'ᠦᠭᠡᠢ', 'ᠤᠲᠠᠰᠤᠨ', 'ᠤ᠋', 'ᠪᠠᠨ', 'ᠪᠦᠷᠢᠶᠡᠰᠦ', 'ᠶ᠋ᠢ', 'ᠦᠵᠦᠭᠦᠷ', 'ᠡᠴᠡ', 'ᠨᠢ', 'ᠨᠢᠭᠡ', 'ᠳᠠᠬᠢᠪᠠᠯ', 'ᠴᠣᠯᠮᠣᠨ', 'ᠮᠢᠨᠦ', 'ᠡᠮᠦᠨ\u180eᠡ', 'ᠶᠠᠮᠠᠷ', 'ᠬᠠᠷᠠᠴᠠ', 'ᠪᠠᠷ', 'ᠲᠡᠭᠦᠨ', 'ᠢ᠋ᠶ᠋ᠡᠨ', 'ᠬᠠᠷᠠᠬᠤ', 'ᠶ᠋ᠢ', 'ᠨᠢ', 'ᠠᠵᠢᠭᠯᠠᠶ\u180eᠠ', 'ᠭᠡᠵᠦ', 'ᠪᠢ', 'ᠳᠣᠲᠣᠷ\u180eᠠ', 'ᠪᠠᠨ', 'ᠪᠣᠳᠣᠪᠠ']
    d = ['ᠴᠠᠬᠢᠯᠭᠠᠨ', 'ᠲᠠᠢ', 'ᠪᠠ', 'ᠴᠠᠬᠢᠯᠭᠠᠨ', 'ᠦᠭᠡᠢ', 'ᠤᠲᠠᠰᠤᠨ', 'ᠤ᠋', 'ᠪᠠᠨ', 'ᠪᠦᠷᠢᠶᠡᠰᠦ', 'ᠶ᠋ᠢ', 'ᠦᠵᠦᠭᠦᠷ', 'ᠡᠴᠡ', 'ᠨᠢ', 'ᠨᠢᠭᠡ', 'ᠢᠨᠴᠠᠭᠠᠪᠠ', 'ᠤᠭᠲᠤᠯ', 'ᠳᠠᠬᠢᠪᠠᠯ', 'ᠴᠣᠯᠮᠣᠨ', 'ᠮᠢᠨᠦ', 'ᠡᠮᠦᠨ\u180eᠡ', 'ᠶᠠᠮᠠᠷ', 'ᠬᠠᠷᠠᠴᠠ', 'ᠪᠠᠷ', 'ᠲᠡᠭᠦᠨ', 'ᠢ᠋ᠶ᠋ᠡᠨ', 'ᠬᠠᠷᠠᠬᠤ', 'ᠶ᠋ᠢ', 'ᠨᠢ', 'ᠠᠵᠢᠭᠯᠠᠶ\u180eᠠ', 'ᠭᠡᠵᠦ', 'ᠪᠢ', 'ᠳᠣᠲᠣᠷ\u180eᠠ', 'ᠪᠠᠨ', 'ᠪᠣᠳᠣᠪᠠ']
    a = [1, 2, 3, 5, 4]
    b = [1, 2, 4, 5]
    erm = ErrorRateMetric()
    print(erm.one_utt_cer(c, d))
    print(erm.all_cer())
    print(erm.one_utt_cer(a, b))
    print(erm.all_cer())

