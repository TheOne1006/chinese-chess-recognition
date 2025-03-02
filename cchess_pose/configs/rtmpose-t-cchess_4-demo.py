"""
Demo config for test
"""
_base_ = [
    "rtmpose-t-cchess_4.py",
]

max_epochs = 10
val_interval = 2
train_cfg = {'max_epochs': max_epochs, 'val_interval': val_interval}

train_batch_size=32
stage2_num_epochs = 2

base_lr=2e-3
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
auto_scale_lr = dict(base_batch_size=36)
