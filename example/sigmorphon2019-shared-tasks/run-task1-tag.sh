#!/bin/bash
arch=$1
pair=$2
highreslang=$(echo $pair | sed -e 's/--/:/g' | cut -d: -f1)
lowreslang=$(echo $pair | sed -e 's/--/:/g' | cut -d: -f2)
dir=example/sigmorphon2019-shared-tasks/sample
python src/train.py \
    --dataset sigmorphon19task1 \
    --train $dir/task1/$pair/$highreslang-train-high $dir/task1/$pair/$lowreslang-train-low \
    --dev $dir/task1/$pair/$lowreslang-dev \
    --model model/sigmorphon19/task1/tag-$arch/$pair --seed 0 \
    --embed_dim 200 --src_hs 400 --trg_hs 400 --dropout 0.4 \
    --src_layer 2 --trg_layer 1 --max_norm 5 --indtag \
    --arch $arch --estop 1e-8 --epochs 1000 --bs 20 --shuffle --patience 10
