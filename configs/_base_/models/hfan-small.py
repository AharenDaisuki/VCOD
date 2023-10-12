#########
# model #
#########

is_linux = False
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255
)
model = dict(
    type = 'EncoderDecoder',
    data_preprocessor = data_preprocessor,
    # pretrained='/hfan/checkpoint/mit_b1.pth',       # set your path
    pretrained = 'C:\\Users\\xyli45\\Desktop\\git\\HFAN\\checkpoint\\mit_b1.pth' if not is_linux 
                 else 'checkpoint/mit_b1.pth',
    backbone=dict( # TODO: DONE
        type='HFANVOS',
        ori_type='mit_b1',
        style='pytorch'),
    decode_head=dict( # TODO: DONE
        type='HFANVOS_Head',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        select_method='hfan',     # hfan
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=256),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
    # test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(271, 271))
)