#!/bin/bash
arch=$1
res=${2:-high}
lang=$3
seed=${4:-0}
# for lang in $(cat data/conll2017/lang.txt); do
python src/train.py \
    --dataset sigmorphon17task1 \
    --train data/conll2017/all/task1/$lang-train-$res \
    --dev data/conll2017/all/task1/$lang-dev \
    --test data/conll2017/answers/task1/$lang-uncovered-test \
    --model model/sigmorphon17-task1/large/tag-$arch/$lang-$res \
    --init init/sigmorphon17-task1/large/seed-$seed/$lang-$res --seed $seed \
    --embed_dim 200 --src_hs 400 --trg_hs 400 --dropout 0.4 \
    --src_layer 2 --trg_layer 1 --max_norm 5 --nb_sample 4 \
    --arch $arch --gpuid 0 --estop 1e-8 --epochs 50 --bs 20 --indtag
# done
