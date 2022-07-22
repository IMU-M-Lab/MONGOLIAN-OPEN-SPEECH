# -*- encoding: utf-8 -*-
"""
@File       :   word2vec_finetune.py
@Created by :   lx
@Create Time:   2021/11/11 13:08
@Description:   word2vec_finetune
"""

import codecs
from gensim.models import Word2Vec


# def process_data():
#     with codecs.open(data_file, 'r', encoding='utf-8') as fin:
#         with codecs.open(out_file, 'a', encoding='utf-8') as fout:
#             for line in fin:
#                 arr = line.strip().split('\t')
#                 text = arr[3].split(':')[1]
#                 fout.write(text + '\n')

def train(use_word=False):
    context = []
    for s in set_list:
        data_file = '/data/lx/data/aishell/processed/raw_wav/{}/format.cut_word_data'.format(s)
        with codecs.open(data_file, 'r', encoding='utf-8') as fin:
            for line in fin:
                arr = line.strip().split('\t')
                if use_word:
                    cut_word = arr[7].split(':')[1]
                    context.append(cut_word.split(','))
                else:
                    char = arr[4].split(':')[1]
                    context.append(char.split(' '))

    model = Word2Vec.load(word2vec_model)
    print('start train')
    print(len(context))
    model.min_count = 3
    model.build_vocab(context, update=True)
    model.train(context, total_examples=len(context), epochs=epoch)
    model.save(word2vec_model_save)
    print('finish')


if __name__ == '__main__':
    epoch = 20
    word2vec_model = '/data/lx/word2vec/word2vec_wx'
    word2vec_model_save = '/data/lx/word2vec/word2vec_wx_ft_' + str(epoch)
    # model = Word2Vec.load(word2vec_model_save)
    # print(model['当地政府'])
    set_list = [
        # 'test',
        # 'dev',
        'train'
    ]
    train()
