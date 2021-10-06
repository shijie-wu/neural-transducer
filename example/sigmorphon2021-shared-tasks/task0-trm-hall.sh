#!/bin/bash
lang=$1
arch=${2:-tagtransformer}

lr=0.001
scheduler=warmupinvsqr
max_steps=20000
warmup=4000
beta2=0.98       # 0.999
label_smooth=0.1 # 0.0
total_eval=50
bs=400 # 256

# transformer
layers=4
hs=1024
embed_dim=256
nb_heads=4
dropout=${2:-0.3}

ckpt_dir=checkpoints/sig21

case "$lang" in
"sjo" | "tur" | "vro") trn_path=2021Task0/part1/surprise_languages ;;
*) trn_path=2021Task0/part1/development_languages ;;
esac
tst_path=2021Task0/part1/ground-truth

python src/train.py \
    --dataset sigmorphon17task1 \
    --train $trn_path/$lang.hall.train \
    --dev $trn_path/$lang.dev \
    --test $tst_path/$lang.test \
    --model $ckpt_dir/$arch-hall/$lang \
    --decode greedy --max_decode_len 32 \
    --embed_dim $embed_dim --src_hs $hs --trg_hs $hs --dropout $dropout --nb_heads $nb_heads \
    --label_smooth $label_smooth --total_eval $total_eval \
    --src_layer $layers --trg_layer $layers --max_norm 1 --lr $lr --shuffle \
    --arch $arch --gpuid 0 --estop 1e-8 --bs $bs --max_steps $max_steps \
    --scheduler $scheduler --warmup_steps $warmup --cleanup_anyway --beta2 $beta2 --bestacc
