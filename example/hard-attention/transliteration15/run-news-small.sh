#!/bin/bash
arch=$1
for pair in ArEn EnBa EnHi EnJa EnKa EnKo EnPe EnTa EnTh JnJk ThEn; do
    python src/train.py \
        --dataset news15 \
        --train data/transliteration15/$pair/train.xml \
        --dev data/transliteration15/$pair/dev.xml \
        --model model/transliteration15/small/$arch/$pair \
        --embed_dim 100 --src_hs 200 --trg_hs 200 --dropout 0.2 \
        --arch $arch --gpuid 0 --estop 1e-8 --epochs 50 --bs 50
done
