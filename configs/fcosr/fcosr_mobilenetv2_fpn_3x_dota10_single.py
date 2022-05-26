_base_ = [
    '../_base_/datasets/dotav1.py', '../_base_/schedules/schedule_3x.py',
    '../_base_/default_runtime.py'
]

data = dict(val=dict(type='DOTAFullValDataset'))

image_size = (1024, 1024)
batch_acc = 4

model = dict(
    type='FCOSR',
    backbone=dict(
        type='MobileNetV2',
        out_indices=(2, 4, 7),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://mmdet/mobilenet_v2')),
    neck=dict(
        type='FPN',
        in_channels=[32, 96, 1280],
        out_channels=128,
        start_level=0,
        num_outs=5,
        add_extra_convs='on_output',
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='FCOSRboxHead',
        num_classes=15,
        in_channels=128,
        feat_channels=256,
        stacked_convs=4,
        strides=(8, 16, 32, 64, 128),
        regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                        (512, 100000000.0)),
        conv_cfg=dict(type='Conv2d'),
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        assigner=dict(
            type='GaussianAssigner',
            gauss_factor=12.0,
            inside_ellipsis_thresh=0.23,
            epsilon=1e-9),
        cls_loss=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            reduction='mean',
            loss_weight=1.0),
        cls_scores='iou',
        reg_loss=dict(type='ProbiouLoss', mode='l1', loss_weight=1.0),
        reg_weights='iou',
        init_cfg=dict(
            type='Normal',
            layer='Conv2d',
            std=0.01,
            override=dict(
                type='Normal',
                name='cls_logits_conv',
                std=0.01,
                bias_prob=0.01))),
    train_cfg=dict(gamma=2.0, alpha=0.25),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=8,
        score_thr=0.1,
        nms=0.1,
        max_per_img=2000,
        rotations=[]))

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

optimizer_config = dict(
    type='GradientCumulativeOptimizerHook',
    cumulative_iters=batch_acc,
    grad_clip=dict(max_norm=35, norm_type=2))

checkpoint_config = dict(interval=1)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'work_dirs/DOTA10/FCOSR-S/FCOSR_mobilenetv2_fpn_3x_dota10_single'
find_unused_parameters = True
load_from = None
resume_from = None
workflow = [('train', 1)]
gpu_ids = range(0, 1)
