_base_ = [
    "swinv2-nano_cchess16-256.py",
]

max_epochs = 50
base_lr = 1e-4

optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=base_lr * 128 * 1 / 512,
        weight_decay=0.01, # 0.01 - 0.03
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
    dict(type='CosineAnnealingLR', eta_min=1e-5, by_epoch=True, end=max_epochs)
]

train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=10)

# load_from = 'work_dirs/swinv2-nano_cchess16-256/epoch_200.pth'

resume = True
