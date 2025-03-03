# defaults to use registries in mmpretrain
default_scope = 'mmpretrain'

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,

    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),

    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type='IterTimerHook'),

    # print log every 10 iterations.
    logger=dict(type='LoggerHook', interval=10),

    # enable the parameter scheduler.
    param_scheduler=dict(type='ParamSchedulerHook'),

    # save checkpoint per epoch.
    checkpoint=dict(
        type='CheckpointHook', 
        save_best='multi-label/mAP',
        rule='greater',
        max_keep_ckpts=5,
        interval=10),

    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type='DistSamplerSeedHook'),

    # 早停
    early_stopping=dict(
        type='EarlyStoppingHook',
        monitor='multi-label/mAP',
        patience=50,
        min_delta=0.001,
        rule='greater'
    ),

    # validation results visualization, set True to enable it.
    visualization=dict(type='VisualizationHook', enable=False, interval=100),
)

vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(type='CChessVisualizer', vis_backends=vis_backends)

