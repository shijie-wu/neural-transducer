#!/bin/bash
arch=$1
lang=$2
res=high
python src/train.py \
    --dataset sigmorphon17task1 \
    --train data/conll2017/all/task1/$lang-train-$res \
    --dev data/conll2017/all/task1/$lang-dev \
    --test data/conll2017/all/task1/$lang-covered-test \
    --model model/sigmorphon17-task1/large/$arch/$lang-$res \
    --embed_dim 200 --src_hs 400 --trg_hs 400 --dropout 0.4 \
    --src_layer 2 --trg_layer 1 --max_norm 5 --nb_sample 4 \
    --arch $arch --gpuid 0 --estop 1e-8 --epochs 50 --bs 20
