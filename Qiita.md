# はじめに
multi GPUで学習するためにpytorchのDistributedDataParallel(DDP)を使ってみたいと思います。  
kaggleの自然言語コンペではhuggingfaceのtransformersを使い自然言語モデルをfine-tuningすることがよくあります。  
そこで、kaggleで使っていくことを考えて、huggingfaceのtutorialである[Fine-tune a pretrained model](https://huggingface.co/docs/transformers/training)をmulti GPUでやっていきたいと思います。

今回実行するプログラムは[ddp_training](https://github.com/oriki101/ddp_training)に保存しています。  
今回使ったdocker imageも公開しているのでオンプレで実行したい方は利用してください([kaggle_pytorch_docker](https://github.com/oriki101/kaggle_pytorch_docker))。

# 1.データの準備
## データのダウンロード
fine-tuning用のデータをロードします。    
利用するデータはYelp Revierwsです。

```python
from datasets import load_dataset

dataset = load_dataset("yelp_review_full")
dataset['train'][0]
```

## Pytorchで扱えるように変換
テキスト情報をtokenizerを利用してトークン化します。  
また、学習時間短縮のためデータを一部のみを取り出します。
```python
from transformers import AutoTokenizer
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
```

# 2.DDPを利用して学習
本題のDDPを利用したmulti GPUでの学習に取り掛かります。  
DDPでの学習は公式の[ドキュメント](https://pytorch.org/docs/master/notes/ddp.html)や[チュートリアル](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)が公開されています。  
これらを参考にhuggingface tutorialである[Fine-tune a pretrained model](https://huggingface.co/docs/transformers/training)を学習するための関数を作成しました。    
学習用の関数全体は以下のようになります(要素の詳細はさらに下に記載していきます)。  
```python
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

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
    train_sampler = DistributedSampler(small_train_dataset,num_replicas=world_size,rank=rank,shuffle=True)
    train_loader = DataLoader(small_train_dataset,batch_size=32,shuffle=train_sampler is None,pin_memory=True,sampler=train_sampler)

    ## optizerとschedulerの定義
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = 3
    num_training_steps = num_epochs * len(train_loader) / world_size
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
```
ここからはtrain関数内でDDPに関わる部分に関して見ていきます。

```python
dist.init_process_group("nccl", rank=rank, world_size=world_size)
```
DDPの利用にはtorch.distributed(as dist)を初期化する必要あります。  
なぜDDPを利用するためにtorch.distributedの初期化が必要なのかを少し調べてみました。  
torch.distributedは[公式ドキュメント](torch.distributedはpytorchで並列処理するのに利用されるもので)によると、複数ノードでの並列処理のための機能を提供します。  
DDPはこの機能に基づきmulti GPUでの分線学習を提供しているため、初期化が必要となるみたいです

<br>

```python
model = DDP(model, device_ids=[rank])
```
この処理でDDP用のモデルを作成します。  
`rank`は並列されたプロセスの番号(0~world_size-1まである)です。  
今回はrankと同じ番号のGPUにモデルを乗せるようにしています。

<br>

```python
train_sampler = DistributedSampler(small_train_dataset,num_replicas=world_size,rank=rank,shuffle=True)
train_loader = DataLoader(small_train_dataset,batch_size=32,shuffle=train_sampler is None,pin_memory=True,sampler=train_sampler)
```
データの読み込みをデータセットのサブセットに制限するサンプラーを作成します。
このサンプラーによってプロセスごとにデータをうまく配分してくれます。
作成したサンプラーをDataLoaderのサンプラーとして渡すことで元のデータセットのサブセットだけを読み出すことができるようになります。

<br>

```python
train_sampler.set_epoch(epoch)
```
DataLoaderのイテレータを作成する前に、各エポックの先頭で set_epoch() メソッドを呼び出すことが、複数のエポックにわたってシャッフリングを適切に動作させるために必要です。   
これをしないと、常に同じ順序が使用されることになります。

<br>

## 学習の実行
```python
# DDPを利用するには環境変数MASTER_ADDRとMASTER_ADDRを設定する必要がある
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"

n_gpus = 4
world_size = n_gpus
mp.spawn(train,
    args=(world_size,small_train_dataset,),
    nprocs=world_size,
    join=True)
```
mp.spawnは関数を実行する複数のプロセスを起動します。  
第一引数に実行する関数、argsに関数の引数、nprocsに並列で実行する数を指定します。  


# 参考
公式ドキュメント以外に参考にしたものは下記のものです。  
とても参考になりました。

- [pytorch DistributedDataPartallel事始め](https://qiita.com/meshidenn/items/1f50246cca075fa0fce2)
- [PyTorchでの分散学習時にはDistributedSamplerを指定することを忘れない！](https://qiita.com/triwave33/items/546d0666dadc8cc51942#distributedsampler)