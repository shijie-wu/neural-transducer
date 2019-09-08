#!/bin/bash
lang=$1
arch=hmm
dir=example/sigmorphon2019-shared-tasks/sample
python src/train.py \
    --dataset sigmorphon19task2 \
    --train $dir/task2/$lang-um-train.conllu.baseline.pred \
    --dev $dir/task2/$lang-um-dev.conllu.baseline.pred \
    --model model/sigmorphon19/task2/monotag-$arch/$lang --seed 0 \
    --embed_dim 200 --src_hs 400 --trg_hs 400 --dropout 0.4 \
    --src_layer 2 --trg_layer 1 --max_norm 5 --indtag \
    --arch $arch --estop 1e-8 --epochs 50 --bs 20 --mono --bestacc
