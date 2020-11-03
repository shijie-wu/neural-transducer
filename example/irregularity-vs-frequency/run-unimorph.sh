#!/bin/bash
arch=hmm
lang=$1
split=$2
seed=0
python src/train.py \
    --dataset unimorph \
    --train data/unimorph/$lang/split/$lang.$split.train \
    --dev data/unimorph/$lang/split/$lang.$split.dev \
    --test data/unimorph/$lang/split/$lang.$split.test \
    --model model/unimorph/large/monotag-$arch/$lang-$split \
    --seed $seed \
    --embed_dim 200 --src_hs 400 --trg_hs 400 --dropout 0.4 \
    --src_layer 2 --trg_layer 1 --max_norm 5 --nb_sample 4 \
    --arch $arch --gpuid 0 --estop 1e-8 --epochs 50 --bs 20 --indtag --mono
