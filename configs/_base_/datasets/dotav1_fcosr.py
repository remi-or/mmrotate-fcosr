# dataset settings
image_size = (1024, 1024)
angle_version = 'oc'
dataset_type = 'DOTADataset'
data_root = 'C:/Users/meri2/Documents/DOTA_1.0/split_ms_dota/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=image_size),
    dict(type='RRandomFlip', flip_ratio=0.5, version=angle_version),
    # dict(type='PolyDiscreteRotate', rotate_ratio=0.5, angles=[0, 90, 180, -90], auto_bound=True, version=angle_version),
    # dict(type='PolyDiscreteRotate', rotate_ratio=0.5, angles=[30, 60], auto_bound=True, version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32), # not OG
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'trainval/annfiles/',
        img_prefix=data_root + 'trainval/images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'trainval/annfiles/',
        img_prefix=data_root + 'trainval/images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test/images/',
        img_prefix=data_root + 'test/images/',
        pipeline=test_pipeline))