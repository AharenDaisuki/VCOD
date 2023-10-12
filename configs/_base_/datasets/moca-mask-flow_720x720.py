####################
# dataset settings #
####################

is_linux = False
dataset_type = 'MoCAMaskFlowDataset' # TODO: 
data_root = 'datasets/MoCA-Mask-Flow/' if is_linux \
            else 'C:\\Users\\xyli45\\Desktop\\datasets\\MoCA-Mask-Flow\\' 
crop_size = (720, 720)

# TODO: training pipeline
train_pipeline = [
    # dict(type='LoadImagesFromFile'),
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    # dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(
        type='RandomResize',
        scale=(1280, 720),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortionMultiImages'),
    dict(type='PhotoMetricDistortion'),
    # dict(type='NormalizeMultiImages', **img_norm_cfg),
    # dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=0),
    # dict(type='DefaultFormatBundle'),
    # dict(type='Collect', keys=['img', 'gt_semantic_seg']),
    dict(type='PackSegInputs')
]

# TODO: testing pipeline
# test_pipeline = [
#     dict(type='LoadImagesFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(2048, 512),
#         # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
#         flip=False,
#         transforms=[
#             # dict(type='AlignedResize', keep_ratio=True, size_divisor=32),
#             dict(type='Resize', keep_ratio=True),
#             dict(type='ResizeToMultiple', size_divisor=32),
#             dict(type='RandomFlip'),
#             dict(type='NormalizeMultiImages', **img_norm_cfg),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img']),
#         ])
# ]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1280, 720), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [dict(type='Resize', scale_factor=r, keep_ratio=True)
             for r in img_ratios],
            [dict(type='ResizeToMultiple', size_divisor=32)], # TODO: ?
            [dict(type='RandomFlip', prob=0., direction='horizontal'),
             dict(type='RandomFlip', prob=1., direction='horizontal')], 
            [dict(type='LoadAnnotations')], 
            [dict(type='PackSegInputs')]
        ])
]

# training data
train_dataloader = dict(
    batch_size = 8,
    num_workers = 8, 
    persistent_workers=True, 
    sampler = dict(type='DefaultSampler', shuffle=True), 
    batch_sampler=dict(type='AspectRatioBatchSampler'), # ?
    dataset=dict(
        type=dataset_type, 
        data_root=data_root, 
        pipeline = train_pipeline,
        img1_dir='frame\\train' if not is_linux else 'frame/train/',
        img2_dir='flow\\train' if not is_linux else 'flow/train/',
        ann_dir='mask\\train' if not is_linux else 'mask/train/',
    )
)

# testing data
val_dataloader = dict(
    batch_size = 8,
    num_workers = 8, 
    persistent_workers=True,
    sampler = dict(type='DefaultSampler', shuffle=True), 
    batch_sampler=dict(type='AspectRatioBatchSampler'), # ?
    dataset=dict(
        type=dataset_type, 
        data_root=data_root, 
        pipeline = test_pipeline,
        img1_dir='frame\\val' if not is_linux else 'frame/val/',
        img2_dir='flow\\val' if not is_linux else 'flow/val/',
        ann_dir='mask\\val' if not is_linux else 'mask/val/',
    )
)

test_dataloader = val_dataloader

# TODO: evaluator
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator