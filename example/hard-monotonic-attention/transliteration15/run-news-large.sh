#!/bin/bash
arch=$1
seed=${2:-0}
for pair in ArEn EnBa EnHi EnJa EnKa EnKo EnPe EnTa EnTh JnJk ThEn; do
    python src/train.py \
        --dataset news15 \
        --train data/transliteration15/$pair/train.xml \
        --dev data/transliteration15/$pair/dev.xml \
        --model model/transliteration15/large/$arch/$pair \
        --init init/transliteration15/large/seed-$seed/$pair --seed $seed \
        --embed_dim 200 --src_hs 400 --trg_hs 400 --dropout 0.4 \
        --src_layer 2 --trg_layer 1 --max_norm 5 --nb_sample 4 \
        --arch $arch --gpuid 0 --estop 1e-8 --epochs 50 --bs 50
done
