#!/bin/bash
#SBATCH --job-name=notags-trm
#SBATCH --mem=10g
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:0:0
#SBATCH --requeue
#SBATCH --partition=gpuk80
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
if [[ -z $CUDA_VISIBLE_DEVICES ]]; then
    gpu=0
else
    gpu=$CUDA_VISIBLE_DEVICES
fi
if [[ $(hostname -f) = *clsp* ]]; then
    export PATH=$HOME/local/app/miniconda3/bin:$PATH
    gpu=`free-gpu`
    # data_dir=/export/b05/shijie/data/sigmorphon/2020/task0
    # ckpt_dir=/export/b05/shijie/checkpoints/sigmorphon20/task0
elif [[ $(hostname -f) = *home* ]]; then
    data_dir=/bigdata/dataset/wikipron-unimorph/split
    ckpt_dir=/bigdata/checkpoints/wikipron-unimorph/v2
else
    data_dir=/home-4/swu53@jhu.edu/data/shijie/wikipron-unimorph/v1
    ckpt_dir=/home-4/swu53@jhu.edu/scratch/experiments/wikipron-unimorph/v1
fi

lang=$1
arch=tagtransformer

lr=0.001
scheduler=warmupinvsqr
max_steps=20000
warmup=4000
beta2=0.98 # 0.999
label_smooth=0.1 # 0.0
total_eval=50
bs=400 # 256

# transformer
layers=4
hs=1024
embed_dim=256
nb_heads=4
dropout=${2:-0.3}


CUDA_VISIBLE_DEVICES=$gpu python src/train.py \
    --dataset wikiphon_unimorph_notag \
    --train $data_dir/$lang.trn \
    --dev $data_dir/$lang.dev \
    --test $data_dir/$lang.tst \
    --model $ckpt_dir/notags-$arch/$lang \
    --embed_dim $embed_dim --src_hs $hs --trg_hs $hs --dropout $dropout --nb_heads $nb_heads \
    --label_smooth $label_smooth --total_eval $total_eval \
    --src_layer $layers --trg_layer $layers --max_norm 1 --lr $lr --shuffle \
    --arch $arch --gpuid 0 --estop 1e-8 --bs $bs --max_steps $max_steps \
    --scheduler $scheduler --warmup_steps $warmup --cleanup_anyway --beta2 $beta2 --bestacc
