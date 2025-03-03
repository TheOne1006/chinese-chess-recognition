_base_ = ['runtime.py']

# 数据集类型及路径
dataset_type = 'CChessDatasetPose4'
data_mode = 'topdown'
data_root = 'data/coco_4/'

dataset_info = {
    'dataset_name':'coco_4',
    'classes':'cchess',
    'paper_info':{
        'author':'theone',
        'title':'coco_4',
        'container':'OpenMMLab',
        'year':'2024',
        'homepage':''
    },
    'keypoint_info':{
        0:{'name':'A0','id':0,'color':[255,0,0],'type': 'upper','swap': ''},
        1:{'name':'A8','id':1,'color':[0,255,0],'type': 'upper','swap': ''},
        2:{'name':'J0','id':2,'color':[100,0,155],'type': 'lower','swap': ''},
        3:{'name':'J8','id':3,'color':[0,0,255],'type': 'lower','swap': ''},

    },
    'skeleton_info': {
        0: {'link':('A0','A8'),'id': 0,'color': [100,150,200]},
        1: {'link':('A0','J0'),'id': 1,'color': [200,100,150]},
        2: {'link':('J0','J8'),'id': 2,'color': [150,120,100]},
        3: {'link':('J8','A8'),'id': 3,'color': [0,120,100]},
    }
}

# 获取关键点个数
NUM_KEYPOINTS = len(dataset_info['keypoint_info'])
dataset_info['joint_weights'] = [1.0] * NUM_KEYPOINTS
dataset_info['sigmas'] = [0.025] * NUM_KEYPOINTS

# 训练超参数
max_epochs = 220 # 训练 epoch 总数
val_interval = 10 # 每隔多少个 epoch 保存一次权重文件
train_cfg = {'max_epochs': max_epochs, 'val_interval': val_interval}
train_batch_size = 78
val_batch_size = 8
stage2_num_epochs = 30
base_lr = 2e-3
randomness = dict(seed=42)

# 优化器
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0.0, 
        bias_decay_mult=0.0, 
        bypass_duplicate=True,
    )
)

# 学习率
param_scheduler = [
    dict(begin=0, by_epoch=False, end=500, start_factor=1e-05, type='LinearLR'),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.01,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=1024)

# codec settings
codec = dict(
    type='SimCCLabel',
    input_size=(256, 256),
    sigma=(3, 3),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)

# 不同输入图像尺寸的参数搭配
# input_size=(256, 256),
# sigma=(12, 12)
# in_featuremap_size=(8, 8)
# input_size可以换成 256、384、512、1024，三个参数等比例缩放
# sigma 表示关键点一维高斯分布的标准差，越大越容易学习，但精度上限会降低，越小越严格，对于人体、人脸等高精度场景，可以调小，RTMPose 原始论文中为 5.66

# 不同模型的 config： https://github.com/open-mmlab/mmpose/tree/dev-1.x/projects/rtmpose/rtmpose/body_2d_keypoint

# 模型：RTMPose-S
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        _scope_='mmdet',
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.167,
        widen_factor=0.375,
        out_indices=(4, ),
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint='https://download.openmmlab.com/mmdetection/v3.0/'
            'rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e-3a2dd350.pth'  # noqa
        )),
    head=dict(
        type='RTMCCHead2',
        in_channels=384,
        out_channels=NUM_KEYPOINTS,
        input_size=codec['input_size'],
        # in_featuremap_size=(8, 8),
        in_featuremap_size=tuple([s // 32 for s in codec['input_size']]),
        simcc_split_ratio=codec['simcc_split_ratio'],
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=128,
            s=64,
            expansion_factor=2,
            dropout_rate=0.,
            drop_path=0.,
            act_fn='SiLU',
            use_rel_bias=False,
            pos_enc=False),
        loss=dict(
            type='KLDiscretLoss',
            use_target_weight=True,
            beta=16., # 越大 训练会更加严格，模型会被迫学习更精确的分布
            label_softmax=True #  softmax 容错
            ),
        loss2=dict(
            type='SoftWeightSmoothL1Loss',
            use_target_weight=True,
            # supervise_empty = True 是 包括了 不可见，权重为零的样本点
            supervise_empty=True,
            beta=1 / 9,
            # loss_weight=0.002, 
        ),
            
        # loss=dict(
        #     type='MultipleLossWrapper',
        #     losses=[
        #         dict(
        #             type='KLDiscretLoss',
        #             use_target_weight=True,
        #             beta=12.,
        #             label_softmax=True),
        #         dict(
        #             type='MSELoss',
        #             use_target_weight=True,
        #             loss_weight=0.5)
        #     ]),

        decoder=codec),
    test_cfg=dict(flip_test=False))


backend_args = dict(backend='local')

# pipelines
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='CopyParseWithPose4', prob=0.5),
    dict(type='RandomUseFullImg', prob=0.5),
    dict(type='RandomGetBBoxCenterScale', paddings=[1.0, 1.1, 1.2, 1.3, 1.4]),
    dict(type='CChessRandomFlip4', direction='horizontal', prob=0.35),
    dict(type='CChessRandomFlip4', direction='vertical', prob=0.35),
    dict(type='RandomHalfCChess4'),
    dict(type='RandomPerspectiveTransform',
        prob=0.6, 
        scale=(0.05, 0.2)
    ),
    # 增大 随机旋转
    dict(type='RandomBBoxTransform', 
         scale_factor = (0.6, 1.8),
         rotate_factor=180),
    dict(type='TopdownAffine', input_size=codec['input_size']),

   dict(
    type='Albumentation',
    transforms=[
        dict(type='Blur', p=0.05),
        dict(type='MedianBlur', p=0.1),
        # dict(type='ChannelShuffle', p=0.15), # 关注线条 和 将 和 帅区分上下
        dict(type='CLAHE', p=0.5),
        dict(type='Downscale', scale_min=0.7, scale_max=0.9, p=0.2),
        dict(type='ColorJitter',
                brightness=0.2,    # 从 0.2 降到 0.1
                contrast=0.2,      # 从 0.2 降到 0.1
                saturation=0.2,    # 从 0.2 降到 0.1
                hue=0.1,         # 从 0.1 降到 0.05
                p=0.3),
        dict(
            type='CoarseDropout',
            max_holes=4,
            max_height=0.25,
            max_width=0.25,
            min_holes=1,
            min_height=0.1,
            min_width=0.1,
            # 填充 114
            fill_value=114,
            p=0.5),
    ]),


    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='RandomUseFullImg', prob=1.0),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

train_pipeline_stage2 = [
    dict(type='RandomUseFullImg', prob=1),
    dict(type='RandomGetBBoxCenterScale', paddings=[1.0, 1.1, 1.2, 1.3, 1.4]),
    dict(type='CChessRandomFlip4', direction='horizontal', prob=0.35),
    dict(type='CChessRandomFlip4', direction='vertical', prob=0.35),
    dict(
        type='RandomBBoxTransform',
        shift_factor=0.25,
        rotate_factor=180,
        scale_factor=(0.7, 1.3)),

    dict(type='TopdownAffine', input_size=codec['input_size']),
    # dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.05),
            dict(type='MedianBlur', p=0.05),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=0.5),
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

# data loaders
train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dataset_info,
        data_mode=data_mode,
        ann_file='annotations/keypoints_train.json',
        data_prefix=dict(img='keypoints_train/'),
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=val_batch_size,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        test_mode=True,
        type=dataset_type,
        data_root=data_root,
        metainfo=dataset_info,
        data_mode=data_mode,
        ann_file='annotations/keypoints_val.json',
        data_prefix=dict(img='keypoints_val/'),
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

# default_hooks = {
#     'checkpoint': {'save_best': 'PCK','rule': 'greater','max_keep_ckpts': 2},
#     'logger': {'interval': 1},
#     #  早停
#     'early_stopping': dict(
#         type='EarlyStoppingHook',
#         monitor='PCK',
#         patience=20,  # 如果连续20个epoch验证指标没有提升就停止训练
#         min_delta=0.001,  # 判定为提升的最小变化阈值
#         rule='greater'  # PCK指标是越大越好
#     )
# }

default_hooks = dict(
    checkpoint=dict(save_best='coco/AP', rule='greater', max_keep_ckpts=3),
    logger=dict(interval=1),
    early_stopping=dict(
        type='EarlyStoppingHook',
        monitor='coco/AP',
        patience=30,  # 如果连续20个epoch验证指标没有提升就停止训练
        min_delta=0.001,  # 判定为提升的最小变化阈值
        rule='greater'  # PCK指标是越大越好
    )
)

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]


# evaluators
val_evaluator = [
    dict(type='CocoMetric', 
         score_mode="keypoint",
         ann_file=data_root + 'annotations/keypoints_val.json'),
    dict(type='PCKAccuracy', thr=0.1),
    dict(type='AUC'),
    dict(
        type='NME', 
        norm_mode='keypoint_distance', 
        keypoint_indices=[0, 2]
    ),
]

test_evaluator = val_evaluator

# resume = True
