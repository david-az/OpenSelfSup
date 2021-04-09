CUDA_VISIBLE_DEVICES=0 python train.py \
    /home/user/OpenSelfSup/az_configs/backbones/hrnet_256_bs32_ep300.py \
    --work_dir /media/disk-1/logs/hrnet_256_bs32_ep300 \
    # --resume_from /media/disk-1/logs/r50_256_bs32_ep100/latest.pth