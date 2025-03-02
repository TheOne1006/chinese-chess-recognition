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

dict_cate_weights = {
    'point': .2,
    'other': 1.,
    'red_king': 1.5,
    'red_advisor': 1.,
    'red_bishop': 1.,
    'red_knight': 1.,
    'red_rook': 1.,
    'red_cannon': 1.,
    'red_pawn': .8,
    'black_king': 1.5,
    'black_advisor': 1.,
    'black_bishop': 1.,
    'black_knight': 1., 
    'black_rook': 1., # black_pawn 与之 表现相近
    'black_cannon': 1.,
    'black_pawn': 1.,
}


_base_ = [
    # '../configs/_base_/default_runtime.py',
    'datasets/multi_label_dataset.py',
    'runtime.py'
]

class_weight = [dict_cate_weights[cate] for cate in dict_cate_names.keys()]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SwinTransformerV2',
        # arch='nano',
        # {'embed_dims': 96,
        # 'depths':     [2, 2,  6,  2],
        # 'num_heads':  [3, 6, 12, 24],
        # 'extra_norm_every_n_blocks': 0}
        arch=dict(
            embed_dims=72,
            depths=[2, 2, 6],
            num_heads=[3, 6, 12],
            extra_norm_every_n_blocks=0,
        ),
        out_indices=(2, ),
        # 是否使用绝对位置嵌入
        img_size=256,
        use_abs_pos_embed=True,
        # patch_size=8, # 默认 4
        # window_size=8,
        drop_path_rate=0.1, # 增加正则化
        # pad_small_map=True,
        ),
    # neck=dict(type='GlobalAveragePooling'),
    # head=dict(
    #     type='LinearClsHead',
    #     num_classes=1000,
    #     in_channels=1024,
    #     init_cfg=None,  # suppress the default init_cfg of LinearClsHead.
    #     loss=dict(
    #         type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
    #     cal_acc=False),
    neck=dict(
        type='CChess10_9Neck',
        in_channels=288,  # 输入通道数
        mid_channels=[256, 128],
        num_classes=len(dict_cate_names.keys()),
    ),
    head=dict(
        type='CChessTableHead',
        num_classes=len(dict_cate_names.keys()),
        loss=dict(
            # type='CrossEntropyLoss', 
            # loss_weight=1.0,
            class_weight=class_weight,
            type='LabelSmoothLoss',
            label_smooth_val=0.1,
            mode='original'
        ),
        cal_acc=True,
        topk=(1, 5),
    ),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
    ],
    # train_cfg=dict(augments=[
    #     dict(type='Mixup', alpha=0.8),
    #     dict(type='CutMix', alpha=1.0)
    # ]),
)


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

data_root = 'data/cchess_multi_label_layout'

max_epochs = 200

# 优化器
# optimizer = dict(type='AdamW', lr=1.5e-5, weight_decay=0.02)
# optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)



# from mmpretrain/mmpretrain/configs/_base_/schedules/imagenet_bs1024_adamw_swin.py

base_lr = 4e-4

# from mmengine.optim import CosineAnnealingLR, LinearLR
# from torch.optim import AdamW

# for batch in each gpu is 128, 8 gpu
# lr = 5e-4 * 128 * 8 / 512 = 0.001
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=base_lr * 128 * 1 / 512,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)),
        paramwise_cfg=dict(
            norm_decay_mult=0.1,
            bias_decay_mult=0.1,
            flat_decay_mult=0.1,
            custom_keys={
                '.absolute_pos_embed': dict(decay_mult=0.1),
                '.relative_position_bias_table': dict(decay_mult=0.1)
            }),
)
# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        end=10,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(type='CosineAnnealingLR', eta_min=1e-5, by_epoch=True, begin=80)
]

# origin
# base_lr = 0.004
# optim_wrapper = dict(
#     optimizer=dict(
#         type='AdamW',
#         lr=base_lr,
#         weight_decay=0.025,
#         # eps=1e-8,
#         # betas=(0.9, 0.999)
#         ),
#     paramwise_cfg=dict(
#         # bias_decay_mult=0,
#         # bypass_duplicate=True,
#         # 设置 backbone 和 head 的权重衰减
#         paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)})
#         ),
#     type='OptimWrapper')

# param_scheduler = [
#     # warm up learning rate scheduler
#     dict(
#         type='LinearLR',
#         start_factor=1e-6 / base_lr,
#         by_epoch=True,
#         end=20,
#         # update by iter
#         convert_to_iter_based=True,
#     ),
#     # main learning rate scheduler
#     dict(type='CosineAnnealingLR', eta_min=1e-5, by_epoch=True, begin=20, end=max_epochs)
# ]


# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=10)
val_cfg = dict()
test_cfg = dict()



# fix 少量数据
auto_scale_lr = dict(base_batch_size=1024)


# resume = True
