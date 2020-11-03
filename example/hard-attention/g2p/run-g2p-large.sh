#!/bin/bash
arch=$1
for data in NETtalk CMUDict; do
    python src/train.py \
        --dataset g2p \
        --train data/g2p/$data.train \
        --dev data/g2p/$data.dev \
        --test data/g2p/$data.test \
        --model model/g2p/large/$arch/$data \
        --embed_dim 200 --src_hs 400 --trg_hs 400 --dropout 0.4 \
        --src_layer 2 --trg_layer 1 --max_norm 5 --nb_sample 4 \
        --arch $arch --gpuid 0 --estop 1e-8 --epochs 50 --bs 20
done
