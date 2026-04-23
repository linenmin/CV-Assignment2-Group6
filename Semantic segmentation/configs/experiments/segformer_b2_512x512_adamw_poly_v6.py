custom_imports = dict(
    imports=[
        "ga2_seg.early_stopping",
        "ga2_seg.mmseg_dataset",
        "ga2_seg.mmseg_transforms",
    ],
    allow_failed_imports=False,
)

_base_ = [
    "../dataset/ga2_voc_like.py",
    "../runtime/default_runtime.py",
]

checkpoint = "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8bf70.pth"
data_root = r"D:\BaiduNetdiskWorkspace\Leuven\8th\Computer Vision\assignment\Group2\kul-computer-vision-ga-2-2026"
project_root = r"D:\BaiduNetdiskWorkspace\Leuven\8th\Computer Vision\assignment\Group2\Semantic segmentation"

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file=project_root + r"\data\splits\train.txt",
        data_prefix=dict(
            img_path=data_root + r"\train\img",
            seg_map_path=data_root + r"\train\seg",
        ),
    )
)
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file=project_root + r"\data\splits\val.txt",
        data_prefix=dict(
            img_path=data_root + r"\train\img",
            seg_map_path=data_root + r"\train\seg",
        ),
    )
)
test_dataloader = val_dataloader

model = dict(
    type="EncoderDecoder",
    data_preprocessor=dict(
        type="SegDataPreProcessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=False,
        pad_val=0,
        seg_pad_val=0,
        size=(512, 512),
    ),
    pretrained=None,
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
        norm_cfg=dict(type="LN", eps=1e-6),
        init_cfg=dict(type="Pretrained", checkpoint=checkpoint),
    ),
    decode_head=dict(
        type="SegformerHead",
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=21,
        norm_cfg=dict(type="BN", requires_grad=True),
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    ),
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)

train_cfg = dict(type="IterBasedTrainLoop", max_iters=30000, val_interval=1000)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

optim_wrapper = dict(
    type="AmpOptimWrapper",
    optimizer=dict(type="AdamW", lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            "pos_block": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
            "head": dict(lr_mult=10.0),
        }
    ),
)

param_scheduler = [
    dict(type="LinearLR", start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type="PolyLR", eta_min=0.0, power=1.0, by_epoch=False, begin=1500, end=30000),
]

randomness = dict(seed=42)

custom_hooks = [
    dict(
        type="DelayedEarlyStoppingHook",
        monitor="mIoU",
        rule="greater",
        min_delta=0.1,
        patience=6,
        begin=5,
        strict=False,
    ),
]

work_dir = "./outputs/logs/exp_v6_segformer_b2"
