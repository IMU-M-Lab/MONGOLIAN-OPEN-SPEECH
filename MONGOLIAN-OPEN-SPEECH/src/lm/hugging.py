# -*- encoding: utf-8 -*-
"""
@File       :   hugging.py
@Created by :   lx
@Create Time:   2021/11/18 16:40
@Description:   hugging
"""
from transformers import BertTokenizer, AlbertForMaskedLM
import torch
import codecs
from datasets import load_dataset
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
import csv
import torch.optim as optim
from utils.scheduler import WarmupLR
import logging
from utils.executor import Executor
from dataset.lm_dataset import CollateFunc


def main():
    pretrained = 'voidful/albert_chinese_tiny'
    start_epoch = 0
    num_epochs = 100
    use_cuda = True
    lr = 0.002
    train_batch_size = 8
    cv_batch_size = train_batch_size // 2 if train_batch_size > 1 else 1
    warmup_steps = 25000
    args = {}

    train_csv = '/data/lx/data/aishell/processed/raw_wav/{}/albert_format.data'.format('train')
    dev_csv = '/data/lx/data/aishell/processed/raw_wav/{}/albert_format.data'.format('dev')
    test_csv = '/data/lx/data/aishell/processed/raw_wav/{}/albert_format.data'.format('test')

    train_collect_fn = CollateFunc()
    dataset = load_dataset('csv', data_files={'train': train_csv, 'test': test_csv, 'dev': dev_csv})

    tokenizer = BertTokenizer.from_pretrained(pretrained)

    def tokenize_function(example):
        # return tokenizer(example["labels"], padding=True, truncation=True, return_tensors="pt")
        return tokenizer(example["labels"], truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    # tokenized_datasets = tokenizer(dataset['train']["labels"], padding=True, truncation=True, return_tensors="pt")

    train_dataloader = DataLoader(tokenized_datasets['train'], shuffle=True,
                                  batch_size=train_batch_size, collate_fn=train_collect_fn)
    cv_dataloader = DataLoader(tokenized_datasets['dev'], shuffle=True, batch_size=cv_batch_size)
    model = AlbertForMaskedLM.from_pretrained(pretrained)

    executor = Executor(use_tqdm=True, lm=True)
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = model.to(device)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = WarmupLR(optimizer, warmup_steps=warmup_steps)

    for epoch in range(start_epoch, num_epochs):
        lr = optimizer.param_groups[0]['lr']
        logging.info('Epoch {} TRAIN info lr {}'.format(epoch, lr))
        executor.train(model, optimizer, scheduler, train_dataloader, device,
                       None, args, None)
        total_loss, num_seen_utts = executor.cv(model, cv_dataloader, device,
                                                args)
    model.save_pretrained('save')
    # inputtext = "今天[MASK]情很好"
    #
    # input_ids = tokenizer()
    # # maskpos = tokenizer.encode(inputtext, add_special_tokens=True).index(103)
    # #
    # # input_ids = torch.tensor(tokenizer.encode(inputtext, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
    # outputs = model(input_ids, labels=input_ids)
    # loss, prediction_scores = outputs[:2]
    # logit_prob = softmax(prediction_scores[0, maskpos], dim=-1).data.tolist()
    # predicted_index = torch.argmax(prediction_scores[0, maskpos]).item()
    # predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
    # print(predicted_token, logit_prob[predicted_index])


if __name__ == '__main__':
    main()
