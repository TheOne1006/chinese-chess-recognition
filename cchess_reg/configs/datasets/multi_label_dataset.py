dict_cate_names = {
    'point': '.',
    'other': 'x',
    'red_king': 'K',
    'red_advisor': 'A',
    'red_bishop': 'B',
    'red_knight': 'N',
    'red_rook': 'R',
    'red_cannon': 'C',
    'red_pawn': 'P',
    'black_king': 'k',
    'black_advisor': 'a',
    'black_bishop': 'b',
    'black_knight': 'n',
    'black_rook': 'r',
    'black_cannon': 'c',
    'black_pawn': 'p',
}

# class_weight = [dict_cate_weights[cate] for cate in dict_cate_names.keys()]

# dataset settings
dataset_type = 'MultiLabelDataset'
data_preprocessor = dict(
    num_classes=len(dict_cate_names),
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

img_norm_cfg = dict(
    mean=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255],
    std=[0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255],
    # to_rgb=True,
)
"""
原图 尺寸 h = 500, w = 450
start_xy = (50, 50)
end_xy = (400, 450)
"""
train_pipeline = [
    dict(type='LoadImageFromFile'),
    # 截取
    dict(type='CenterCrop', crop_size=(400, 450)),
    # 混淆处理
    # dict(type='CChessCachedMixUp', 
    #      prob=0.3,
    #      img_scale=(400, 450), 
    #      rotate_angle=(-90, 90)),
    dict(type='CChessMixSinglePngCls',
         img_scale=(400, 450),
         max_mix_cells=15,
         png_resources_path='data/single_cls2_png',
         prob=0.7),

    # # 缩放 到 统一的尺寸
    dict(type='Resize', scale=(280, 315)), # original (280, 315)
    dict(type='CChessCachedCopyHalf', prob=0.3),
    dict(type='CChessHalfFlip', flip_mode='horizontal', prob=0.5),
    dict(type='CChessHalfFlip', flip_mode='vertical', prob=0.5),

    dict(type='CChessRandomFlip', prob=[0.2, 0.2, 0.2], direction=['horizontal', 'vertical', 'diagonal']),

    # dict(type='RandomFlip', prob=0.3, direction='vertical'),

    # Blur
    dict(
        type='GaussianBlur',
        magnitude_range=(0.1, 1.2),
        magnitude_std='inf',
        prob=0.3),

    # 随机增强
    dict(
        type='RandAugment',
        policies=[
                # safe
                dict(type='AutoContrast'), # 自动对比度
                dict(type='Equalize'), # 均衡化
                dict(type='Posterize', magnitude_range=(4, 0)), # 海报化
                # dict(type='Solarize', magnitude_range=(256, 200)), # 曝光 256, 235
                dict(type='SolarizeAdd', magnitude_range=(0, 110)), # 曝光加
                dict(type='ColorTransform', magnitude_range=(-0.2, 0.2)), # 颜色变换
                dict(type='Contrast', magnitude_range=(0.5, 0.9)), # 对比度
                dict(type='Brightness', magnitude_range=(0.5, 0.9)), # 亮度 变化 
                dict(type='Sharpness', magnitude_range=(0, 0.9)), # 锐化


                dict(type='Rotate', magnitude_range=(0, 25)), # 旋转
    
                # 增加小幅度 位移
                dict(type='Shear', magnitude_range=(0, 0.1), direction='horizontal'), # 水平错切
                dict(type='Shear', magnitude_range=(0, 0.1), direction='vertical'), # 垂直错切
                dict(type='Translate', magnitude_range=(0, 0.1), direction='horizontal'), # 水平平移
                dict(type='Translate', magnitude_range=(0, 0.1), direction='vertical'), # 垂直平移
        ],
        num_policies=3,        # 每次随机选择2个操作
        total_level=10,        # 总的增强等级
        magnitude_level=5,     # 当前使用的增强强度
        magnitude_std=0.5,     # 增强强度的随机偏差
        hparams=dict(
            pad_val=[round(x) for x in img_norm_cfg['mean']],
            interpolation='bicubic'
        )
    ),

    # 随机遮罩
    dict(
        type='RandomErasing',
        erase_prob=0.5,
        mode='rand',
        min_area_ratio=0.0025,
        max_area_ratio=0.005,
        fill_color=img_norm_cfg['mean'],
        fill_std=img_norm_cfg['std']),
    dict(
        type='RandomErasing',
        erase_prob=0.8,
        mode='rand',
        min_area_ratio=0.0025,
        max_area_ratio=0.005,
        fill_color=img_norm_cfg['mean'],
        fill_std=img_norm_cfg['std']),
    dict(
        type='ColorJitter',
        brightness=0.2,    # 亮度
        contrast=0.2,      # 对比度
        saturation=0.2,    # 饱和度
        hue=0.12         # 色相
    ),

    # 透视变换
    # dict(type='RandomPerspectiveTransform', 
    #      size_scale=(0.9, 1.1),
    #      scale=(0.02, 0.06),
    #      prob=0.5),

    dict(type='RandomPerspectiveTransform',  # 透视变换，尺度更大一点
        size_scale=(0.7, 1.3),
        scale=(0.05, 0.18), # 大一点
        prob=0.7),

    dict(type='CChessPackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    # 截取
    dict(type='CenterCrop', crop_size=(400, 450)),
    # 缩放 到 统一的尺寸
    dict(type='Resize', scale=(280, 315)), # original (280, 315)
    dict(type='CChessPackInputs'),
]

data_root = 'data/cchess_multi_label_layout'



train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/train.json',
        data_prefix='train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/val.json',
        data_prefix='val',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)


cchess_prices_args = dict(
    # 2 到 15
    filter_gt_labels=list(range(2, 16)),
    prefix='cchess_prices',
)

# 评估器, 
val_evaluator = [
    # dict(type='CChessPrecisionWithLayout'),
    dict(type='CChessPrecisionWith16Class'),
    dict(type='CChessAveragePrecision', average=None),
    dict(type='CChessAveragePrecision', average='macro'),
    dict(type='CChessMultiLabelMetric', average='macro'),  # class-wise mean
    dict(type='CChessMultiLabelMetric', average='micro'),  # overall mean

    # cchess_prices 即 with out 0, 1
    dict(
        type='CChessPrecisionWith16Class', 
        filter_gt_labels=cchess_prices_args['filter_gt_labels'],
        prefix='cchess_prices_14cls',
        ),    
    dict(type='CChessAveragePrecision', average=None, **cchess_prices_args),
    dict(type='CChessAveragePrecision', average='macro', **cchess_prices_args),
    dict(type='CChessMultiLabelMetric', average='macro', **cchess_prices_args),  # class-wise mean
    dict(type='CChessMultiLabelMetric', average='micro', **cchess_prices_args),  # overall mean
    dict(type='CChessFullAccuracy', errK=(1, 3, 5)),
]

# val_evaluator = dict(type='Accuracy', topk=(1, 5))

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator

