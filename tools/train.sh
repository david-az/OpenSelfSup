CUDA_VISIBLE_DEVICES=2 python train.py \
    /home/david/OpenSelfSup/az_configs/r50_256_bs32_ep300_imagenet.py \
    --work_dir /home/david/OpenSelfSup/logs/r50_256_bs32_ep300_imagenet \
    --resume_from /home/david/OpenSelfSup/logs/r50_256_bs32_ep300_imagenet/latest.pth