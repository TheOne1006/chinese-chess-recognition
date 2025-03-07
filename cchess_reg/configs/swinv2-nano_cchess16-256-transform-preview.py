_base_ = [
    "swinv2-nano_cchess16-256.py",
]


train_pipeline = [
    dict(type='LoadImageFromFile'),
    # 截取
    dict(type='CenterCrop', crop_size=(400, 450)),
    dict(type='CChessMixSinglePngCls',
         img_scale=(400, 450),
         max_mix_cells=15,
         png_resources_path='data/single_cls2_png',
         prob=0.99),


    dict(type='CChessPackInputs'),
]
