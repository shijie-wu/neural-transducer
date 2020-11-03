#!/bin/bash
gpu=0
data_dir=task0-data/processed
ckpt_dir=checkpoints/sigmorphon20-task0

arch=hmm
lang=$1

CUDA_VISIBLE_DEVICES=$gpu python src/train.py \
    --dataset sigmorphon17task1 \
    --train $data_dir/$lang.hall.trn \
    --dev $data_dir/$lang.dev \
    --test $data_dir/$lang.tst \
    --model $ckpt_dir/hall-mono-$arch/default/$lang \
    --embed_dim 200 --src_hs 400 --trg_hs 400 --dropout 0.4 \
    --src_layer 2 --trg_layer 1 --max_norm 5 --shuffle \
    --arch $arch --gpuid 0 --estop 1e-8 --epochs 50 --bs 50 --bestacc --indtag --mono
