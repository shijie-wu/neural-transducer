#!/bin/bash
#SBATCH --job-name=tags-mono
#SBATCH --mem=10g
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:0:0
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

arch=hmm
lang=$1


CUDA_VISIBLE_DEVICES=$gpu python src/train.py \
    --dataset wikiphon_unimorph \
    --train $data_dir/$lang.trn \
    --dev $data_dir/$lang.dev \
    --test $data_dir/$lang.tst \
    --model $ckpt_dir/tags-$arch/$lang \
    --embed_dim 200 --src_hs 400 --trg_hs 400 --dropout 0.4 \
    --src_layer 2 --trg_layer 1 --max_norm 5 --shuffle \
    --arch $arch --gpuid 0 --estop 1e-8 --epochs 50 --patience 3 --bs 20 --bestacc --indtag --mono
