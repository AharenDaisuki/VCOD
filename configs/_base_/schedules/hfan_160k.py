####################
# runtime settings #
####################

# optimizer
optimizer = dict(type='AdamW', lr=1e-5, betas=(0.9, 0.999), weight_decay=0.01)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
# learning rate
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=160000,
        by_epoch=False)
]
# schedule
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=160000, val_interval=16000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=16000, save_best=''), # TODO
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook')
)
# visualization
vis_backends = [dict(type='TensorboardVisBackend')]
visualizer = dict(type='Visualizer', vis_backends=vis_backends)
# randomness
randomness=dict(seed=1004, deterministic=True, diff_rank_seed=False)
# misc
log_level = 'INFO'
load_from = None
resume = False