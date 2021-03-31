import copy
_base_ = '/home/david/OpenSelfSup/configs/base.py'

ann_file = '/home/david/AZmed-ai/az_annot_files/samples/s1_ccfrtr.json'

# model settings
model = dict(
    type='BYOL',
    pretrained='torchvision://resnet50',
    base_momentum=0.99,
    pre_conv=True,
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[4]),  # 0: conv-1, x: stage-x
    neck=dict(
        type='NonLinearNeckSimCLR',
        in_channels=2048,
        hid_channels=4096,
        out_channels=256,
        num_layers=2,
        sync_bn=False,
        with_bias=True,
        with_last_bn=False,
        with_avg_pool=True),
    head=dict(type='LatentPredictHead',
              size_average=True,
              predictor=dict(type='NonLinearNeckSimCLR',
                             in_channels=256, hid_channels=4096,
                             out_channels=256, num_layers=2, sync_bn=False,
                             with_bias=True, with_last_bn=False, with_avg_pool=False)))

dataset_type = 'PrefetchImagesDataset'

train_pipeline = [
    dict(type='RandomResizedCrop', size=256, scale=(0.9, 1.), interpolation=3),
    dict(type='RandomHorizontalFlip'),
    dict(type='RandomVerticalFlip'),
    dict(type='RandomRotation', degrees=(-20, 20)),
    dict(type='NormalizeMeanVar')
]

prefetch = None
img_norm_cfg = None
data = dict(
    imgs_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=ann_file,
        transform=train_pipeline,
        prefetch=True,
        prefetch_size=300,
    ))
    
# additional hooks
update_interval = 1  # interval for accumulate gradient
custom_hooks = [
    dict(type='BYOLHook', end_momentum=1., update_interval=update_interval)
]
# optimizer
optimizer = dict(type='Adam', lr=3e-4)

# apex
use_fp16 = False
optimizer_config = dict(update_interval=update_interval, use_fp16=use_fp16)

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-7,
    warmup='linear',
    warmup_iters=10,
    warmup_ratio=0.0001, # cannot be 0
    warmup_by_epoch=True)
checkpoint_config = dict(interval=10)
# runtime settings
total_epochs = 300
