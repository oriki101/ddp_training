# ddp_training
このリポジトリはDDPを利用してmulti GPUで学習をやってみようと思い作成したプログラムを保存したリポジトリです。  
huggingfaceのtutorialである[Fine-tune a pretrained model](https://huggingface.co/docs/transformers/training)をmulti GPUでやってみました。  
実行したことに関する軽い説明はQiitaに投稿しています([huggingface tutorial "Fine-tune a pretrained model"をDDPでやってみた](https://qiita.com/oriki101/items/df58d1f1eff8642fe657))。

dockerを利用して環境構築を行なっています。
今回使ったdocker imageは別のリポジトリで公開しているのでオンプレで実行したい方は利用してください([kaggle_pytorch_docker](https://github.com/oriki101/kaggle_pytorch_docker))。

## quick start
```bash
$ git https://github.com/oriki101/ddp_training.git
$ cd ddp_training/docker
$ ./compose_up.sh
```
localhost:8888か<実行したPCのIPアドレス>:8888にアクセスしてください。  
そしてterminal上で
```bash
$ cd ddp_training/script
$ python ddp_training.py
```
