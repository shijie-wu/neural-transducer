#!/bin/bash
arch=$1
for data in NETtalk CMUDict; do
    python src/train.py \
        --dataset g2p \
        --train data/g2p/$data.train \
        --dev data/g2p/$data.dev \
        --test data/g2p/$data.test \
        --model model/g2p/small/$arch/$data \
        --embed_dim 100 --src_hs 200 --trg_hs 200 --dropout 0.2 \
        --arch $arch --gpuid 0 --estop 1e-8 --epochs 50 --bs 20
done
