"""Inference settings for the existing WHU SegFormer checkpoint.

Network and preprocessing settings retain the checkpoint's embedded training
configuration. Full model weights must be supplied by the caller; backbone
pretraining and dataset/training configuration are not needed for inference.
"""

default_scope = "mmseg"
model = dict(
    type="EncoderDecoder",
    pretrained=None,
    data_preprocessor=dict(
        type="SegDataPreProcessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=(512, 1024),
    ),
    backbone=dict(
        type="MixVisionTransformer",
        in_channels=3,
        embed_dims=64,
        num_stages=4,
        num_layers=[3, 4, 6, 3],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        init_cfg=None,
    ),
    decode_head=dict(
        type="SegformerHead",
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=dict(type="SyncBN", requires_grad=True),
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    ),
    train_cfg=dict(),
    test_cfg=dict(mode="slide", crop_size=(512, 1024), stride=(768, 768)),
)
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(2048, 1024), keep_ratio=True),
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]
# MMSegInferencer reads this pipeline without constructing a dataset loader.
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
visualizer = dict(
    name="visualizer",
    type="SegLocalVisualizer",
    vis_backends=[dict(type="LocalVisBackend")],
)
