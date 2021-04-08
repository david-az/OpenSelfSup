#!/bin/bash

exp_name=r50_jitter_ep100
ssl_config=$HOME/OpenSelfSup/az_configs/augmentations/$exp_name.py
ssl_work_dir=$HOME/OpenSelfSup/logs/$exp_name 

supervised_config=$HOME/AZmed-ai/az_configs/ssl/retinanet_$exp_name.py
supervised_workdir=$HOME/AZmed-ai/logs/retinanet_$exp_name

device=1

CUDA_VISIBLE_DEVICES=$device python train.py \
    $ssl_config \
    --work_dir $ssl_work_dir \
    # --resume_from $ssl_work_dir/latest.pth

python extract_backbone_weights.py $ssl_work_dir/latest.pth \
    $ssl_work_dir/backbone_latest.pth

cat $HOME/AZmed-ai/az_configs/trauma/retinanet_r50.py \
    | sed -e "s|pretrained=None|pretrained='$ssl_work_dir/backbone_latest.pth'|" \
    | sed '/load_from =/c\load_from=None' > $supervised_config

cd $HOME/AZmed-ai/az_sh

CUDA_VISIBLE_DEVICES=$device python ../tools/train.py \
    $supervised_config \
    --work-dir $supervised_workdir \
    --gpu-ids 0 \
    --delete-until-epoch 40 \
    --baseline-url https://www.notion.so/azmed/dd3a66767f8d43a5b29cac811d363492?v=6065f593198a4475963813608ff145b7 \
    --baseline-id 0 \