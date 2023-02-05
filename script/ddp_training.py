#!/usr/bin/env python
# coding: utf-8

import pip
pip.main(['install', 'datasets'])


from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import os


def train(rank, world_size, small_train_dataset):
    # refer to https://pytorch.org/docs/master/notes/ddp.html
    # DDPの利用にはdist.init_process_groupで初期化する必要あり
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # モデルの作成
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
    model = model.to(rank)
    # DDP用のモデルの作成
    model = DDP(model, device_ids=[rank])

    #DDP用のサンプラーの作成
    ## これを使うっことによりサンプルをプロセスごとにうまく配分してくれるらしい
    train_sampler = DistributedSampler(small_train_dataset,
                                                        num_replicas=world_size,
                                                        rank=rank,
                                                        shuffle=True)
    train_loader = DataLoader(small_train_dataset,
                                            batch_size=32,
                                            shuffle=train_sampler is None,
                                            pin_memory=True,
                                            sampler=train_sampler)

    ## optizerとschedulerの定義
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = 3
    #num_training_steps = num_epochs * len(train_loader) / world_size
    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(name='linear', optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    #学習
    progress_bar = tqdm(range(int(num_training_steps + 1)))
    model.train()
    for epoch in range(num_epochs):
        # データの順序を帰るためにepochごとにset_epochをする必要あり
        train_sampler.set_epoch(epoch)
        for batch in train_loader:
            batch = {k: v.to(rank) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)


if __name__=="__main__":
    # 1.データの準備
    ## データのダウンロード
    # fine-tuning用のデータをロードします。    
    # 利用するデータはYelp Revierwsです。
    dataset = load_dataset("yelp_review_full")
    dataset['train'][0]


    ## Pytorchで扱えるように変換
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # text情報はモデルに入力しないため削除
    tokenized_datasets = tokenized_datasets.remove_columns(['text'])
    # モデルでは引数がlabelsであると仮定されているので、labelカラムの名前をlabelsに変更
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    # Pytorchに入力できるようにlistからtorchに変更
    tokenized_datasets.set_format('torch')

    # データ量が多いため一部のみ利用
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(5000))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

    # 2. DDPを利用して学習
    # DDPを利用するには環境変数MASTER_ADDRとMASTER_ADDRを設定する必要がある
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    n_gpus = 4
    world_size = n_gpus
    mp.spawn(train,
        args=(world_size,small_train_dataset,),
        nprocs=world_size,
        join=True)