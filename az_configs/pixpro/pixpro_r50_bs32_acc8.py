import copy
_base_ = '/home/david/OpenSelfSup/configs/base.py'

ann_file = '/home/david/AZmed-ai/az_annot_files/samples/s1_ccfrtr.json'

# model settings
model = dict(
    type='PixPro',
    base_momentum=0.99,
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=1,
        out_indices=[4]))

dataset_type = 'PixProDataset'

train_pipeline = [
        dict(type='RandomResizedCropCoord', size=256, scale=(0.9, 1.)),
        dict(type='RandomHorizontalFlipCoord'),
        dict(type='RandomVerticalFlipCoord'),
        dict(type='RandomRotation', degrees=(-20, 20)),
        dict(type='ToTensor'),
        dict(type='NormalizeMeanVar2')
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
        prefetch_size=300    
    ))

# additional hooks
update_interval = 8  # interval for accumulate gradient
custom_hooks = [
    dict(type='BYOLHook', end_momentum=1., update_interval=update_interval)
]

# optimizer
optimizer = dict(type='LARS', lr=0.1, weight_decay=0.00001, momentum=0.9,
                 paramwise_options={
                    '(bn|gn)(\d+)?.(weight|bias)': dict(weight_decay=0., lars_exclude=True),
                    'bias': dict(weight_decay=0., lars_exclude=True)})

# apex
use_fp16 = False
optimizer_config = dict(update_interval=update_interval, use_fp16=use_fp16)

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-6,
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=0.0001, # cannot be 0
    warmup_by_epoch=True)
checkpoint_config = dict(interval=50)
# runtime settings
total_epochs = 100
