_base_ = [
    "swinv2-nano_cchess16-256.py",
]

# fix 少量数据
auto_scale_lr = dict(base_batch_size=8)


train_dataloader = dict(
    batch_size=4,
    num_workers=2,
)


max_epochs=6
train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=2)


# 训练 6 个 epoch
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        end=max_epochs,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(type='CosineAnnealingLR', eta_min=1e-5, by_epoch=True, begin=1)
]
