# -*- encoding: utf-8 -*-
"""
@File       :   text_process.py
@Created by :   lx
@Create Time:   2021/10/15 16:13
@Description:   对数据集做分词、翻译等操作
"""

import codecs
from google_trans import google_translator, google_new_transError
from multiprocessing.dummy import Pool as ThreadPool
import jieba
import os
import sys
from tqdm import tqdm


class TranslateForASR:
    def __init__(self, data_file, out_file, translator_conf, retry_time=5):
        self.data_file = data_file
        self.out_file = out_file
        self.translator_conf = translator_conf
        self.retry_time = retry_time

        if not os.path.isfile(out_file):
            self.processed_num = 0
            return
        with codecs.open(out_file, 'r', encoding='utf-8') as fout:
            processed_num = 0
            for _ in fout:
                processed_num += 1
        self.processed_num = processed_num
        print('already processed sentence num:' + str(processed_num))

    def _request(self, text):
        t = google_translator(**self.translator_conf)
        for r_t in range(self.retry_time):
            try:
                translate_text = t.translate(text, lang_src='de', lang_tgt='zh-cn')
            except google_new_transError as e:
                if r_t == self.retry_time - 1:
                    raise e
            else:
                return text + '\t' + translate_text

    def translate_parallel(self):
        # 修改过 但没测试过
        pool = ThreadPool(15)  # Threads
        with codecs.open(self.data_file, 'r', encoding='utf-8') as fin:
            texts = []
            for _ in range(self.processed_num):
                fin.readline()
            for line in fin:
                arr = line.strip().split('\t')
                texts.append(arr[3].split(':')[1])
            try:
                result = tqdm(pool.imap(self._request, texts), total=len(texts))
            except Exception as e:
                raise e

        pool.close()
        pool.join()
        print('translate end')

        with codecs.open(self.out_file, 'a+', encoding='utf-8') as fout:
            for line in result:
                fout.write(line)

    def translate_single(self):
        translator = google_translator(**self.translator_conf)
        with codecs.open(self.data_file, 'r', encoding='utf-8') as fin:
            with codecs.open(self.out_file, 'a', encoding='utf-8') as fout:
                for _ in range(self.processed_num):
                    fin.readline()
                for line in tqdm(fin):
                    arr = line.strip().split('\t')
                    text = arr[3].split(':')[1]
                    for r_t in range(self.retry_time):
                        try:
                            result = translator.translate(text, lang_src='de', lang_tgt='zh-cn')
                        except google_new_transError as e:
                            if r_t == self.retry_time - 1:
                                raise e
                            continue
                        else:
                            break
                    fout.write(text + '\t' + result + '\n')


class CutWordForASR:
    def __init__(self, data_file, out_file):
        self.data_file = data_file
        self.out_file = out_file

    def _check(self):
        if not os.path.isfile(self.out_file):
            self.processed_num = 0
            return
        with codecs.open(self.out_file, 'r', encoding='utf-8') as fout:
            processed_num = 0
            for _ in fout:
                processed_num += 1
        self.processed_num = processed_num
        print('already processed sentence num:' + str(processed_num))

    def cut_single(self, check=True):
        if check:
            self._check()
        with codecs.open(self.data_file, 'r', encoding='utf-8') as fin:
            with codecs.open(self.out_file, 'a', encoding='utf-8') as fout:
                for _ in range(self.processed_num):
                    fin.readline()
                for line in tqdm(fin):
                    arr = line.strip().split('\t')
                    token_id = arr[5].split(':')[1]
                    token_id = list(map(int, token_id.split(' ')))
                    text = arr[3].split(':')[1]
                    cut_results = jieba.cut(text)
                    cut_word_text = ",".join(cut_results)
                    cut_word_id = ''
                    cwi = 0
                    for cwt in cut_word_text:
                        strip = '' if cwi == 0 else ' '
                        if cwt ==',':
                            append = ','
                        else:
                            append = strip + str(token_id[cwi])
                            cwi+=1
                        cut_word_id += append
                    fout.write(line[:-2] + '\tcut_word:' + cut_word_text + '\tcut_word_id:' + cut_word_id + '\n')
        self.verify()

    def verify(self):
        # 验证两边文件是否对应正确
        data = []
        with codecs.open(self.data_file, 'r', encoding='utf-8') as f_data:
            for line in f_data:
                arr = line.strip().split('\t')
                text = arr[0].split(':')[1]
                data.append(text)
        with codecs.open(self.out_file, 'r', encoding='utf-8') as f_out:
            for i, line in enumerate(f_out):
                arr = line.strip().split('\t')
                text = arr[0].split(':')[1]
                if not data[i] == text:
                    print('wrong:' + str(i) + ',' + text + ',' + data[i])
            assert len(data) == i + 1
            print('verify success')


def translate():
    data_file = '/data01/lx2/data/aishell/processed/raw_wav/train/format.data'
    out_file = '/data01/lx2/data/aishell/processed/raw_wav/train/format.translate_data'

    # translator_conf参数 参考google_translator
    # prox = {'http': '127.0.0.1:56986', 'https': '127.0.0.1:56986'}
    # translator_conf = {'url_suffix': "cn", 'timeout': 5, 'proxies': prox}
    translator_conf = {'url_suffix': "cn", 'timeout': 5}
    trans_text = TranslateForASR(data_file, out_file, translator_conf)
    trans_text.translate_single()
    # trans_text.translate_parallel()


def cut_word():
    set_list = [
        'test',
        'dev',
        'train'
    ]
    for s in set_list:
        data_file = '/data/lx/data/aishell/processed/raw_wav/{}/format.data'.format(s)
        out_file = '/data/lx/data/aishell/processed/raw_wav/{}/format.cut_word_data'.format(s)
        cut = CutWordForASR(data_file, out_file)
        cut.cut_single()
        # cut.verify()


if __name__ == "__main__":
    # translate()
    cut_word()
