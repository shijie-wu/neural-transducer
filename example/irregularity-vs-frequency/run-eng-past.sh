#!/bin/bash
mode=$1 # odonnell or albright
arch=hmm
seed=0
python src/train.py \
    --dataset unimorph \
    --train data/unimorph/eng.$mode/eng.$mode.train \
    --dev data/unimorph/eng.$mode/eng.$mode.dev \
    --test data/unimorph/eng.$mode/eng.$mode.test \
    --model model/unimorph/large/monotag-$arch/eng.$mode \
    --seed $seed \
    --embed_dim 200 --src_hs 400 --trg_hs 400 --dropout 0.4 \
    --src_layer 2 --trg_layer 1 --max_norm 5 --nb_sample 4 \
    --arch $arch --gpuid 0 --estop 1e-8 --epochs 50 --bs 20 --indtag --mono
