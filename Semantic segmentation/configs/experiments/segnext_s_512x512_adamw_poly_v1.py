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
    "../model/segnext_s_ga2.py",
    "../runtime/default_runtime.py",
]

load_from = "https://download.openmmlab.com/mmsegmentation/v0.5/segnext/segnext_mscan-s_1x16_512x512_adamw_160k_ade20k/segnext_mscan-s_1x16_512x512_adamw_160k_ade20k_20230214_113014-43013668.pth"

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

work_dir = "./outputs/logs/segnext_s_512x512_adamw_poly_v1"
