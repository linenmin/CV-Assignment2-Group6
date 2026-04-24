from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .labels import ALL_CLASSES, PALETTE
from .paths import get_dataset_root, get_project_root


SEG_RESULT_VARIANTS = {
    "b": {
        "backbone": "SegMANEncoder_b",
        "in_channels": [96, 160, 364, 560],
        "channels": 180,
        "feat_proj_dim": 320,
    },
    "l": {
        "backbone": "SegMANEncoder_l",
        "in_channels": [96, 192, 432, 640],
        "channels": 224,
        "feat_proj_dim": 432,
    },
}


@dataclass(frozen=True)
class SegMANExportResult:
    export_root: Path
    train_count: int
    val_count: int
    test_count: int


def get_segman_export_root() -> Path:
    return get_project_root() / "data" / "segman_ga2"


def load_split_ids(split_name: str) -> list[int]:
    split_path = get_project_root() / "data" / "splits" / f"{split_name}.txt"
    return [int(line.strip()) for line in split_path.read_text().splitlines() if line.strip()]


def _save_png(array: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8)).save(output_path)


def _export_train_subset(ids: list[int], subset: str, export_root: Path, overwrite: bool) -> int:
    dataset_root = get_dataset_root()
    image_dir = export_root / "images" / subset
    ann_dir = export_root / "annotations" / subset

    for train_id in ids:
        image_path = image_dir / f"train_{train_id}.png"
        ann_path = ann_dir / f"train_{train_id}.png"
        if overwrite or not image_path.exists():
            image = np.load(dataset_root / "train" / "img" / f"train_{train_id}.npy")
            _save_png(image, image_path)
        if overwrite or not ann_path.exists():
            mask = np.load(dataset_root / "train" / "seg" / f"train_{train_id}.npy")
            _save_png(mask, ann_path)

    split_path = export_root / f"{subset}.txt"
    split_path.write_text("\n".join(f"train_{train_id}" for train_id in ids) + "\n")
    return len(ids)


def _export_test(export_root: Path, overwrite: bool, limit: int | None = None) -> int:
    dataset_root = get_dataset_root()
    test_image_dir = dataset_root / "test" / "img"
    output_dir = export_root / "images" / "test"
    test_paths = sorted(test_image_dir.glob("test_*.npy"), key=lambda path: int(path.stem.split("_")[1]))
    if limit is not None:
        test_paths = test_paths[:limit]

    test_names: list[str] = []
    for input_path in test_paths:
        output_path = output_dir / f"{input_path.stem}.png"
        if overwrite or not output_path.exists():
            _save_png(np.load(input_path), output_path)
        test_names.append(input_path.stem)

    (export_root / "test.txt").write_text("\n".join(test_names) + "\n")
    return len(test_names)


def export_segman_dataset(overwrite: bool = False, test_limit: int | None = None) -> SegMANExportResult:
    export_root = get_segman_export_root()
    train_count = _export_train_subset(load_split_ids("train"), "training", export_root, overwrite)
    val_count = _export_train_subset(load_split_ids("val"), "validation", export_root, overwrite)
    test_count = _export_test(export_root, overwrite, limit=test_limit)
    return SegMANExportResult(
        export_root=export_root,
        train_count=train_count,
        val_count=val_count,
        test_count=test_count,
    )


def build_segman_config_text(
    variant: str,
    export_root: str | Path,
    encoder_checkpoint: str | Path,
    max_iters: int = 30000,
    val_interval: int = 1000,
    batch_size: int = 2,
    workers: int = 4,
) -> str:
    variant = variant.lower()
    if variant not in SEG_RESULT_VARIANTS:
        raise ValueError(f"Unsupported SegMAN variant: {variant}")

    spec = SEG_RESULT_VARIANTS[variant]
    classes = tuple(ALL_CLASSES)
    palette = tuple(tuple(int(value) for value in color) for color in PALETTE)
    export_root = str(Path(export_root).resolve()).replace("\\", "/")
    encoder_checkpoint = str(Path(encoder_checkpoint).resolve()).replace("\\", "/")

    return f'''_base_ = [
    '../../_base_/models/segman.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k_adamw.py',
]

dataset_type = 'CustomDataset'
data_root = r'{export_root}'
classes = {classes!r}
palette = {palette!r}
crop_size = (512, 512)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]),
]

data = dict(
    samples_per_gpu={batch_size},
    workers_per_gpu={workers},
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/training',
        ann_dir='annotations/training',
        img_suffix='.png',
        seg_map_suffix='.png',
        split='training.txt',
        classes=classes,
        palette=palette,
        reduce_zero_label=False,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        img_suffix='.png',
        seg_map_suffix='.png',
        split='validation.txt',
        classes=classes,
        palette=palette,
        reduce_zero_label=False,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        img_suffix='.png',
        seg_map_suffix='.png',
        split='validation.txt',
        classes=classes,
        palette=palette,
        reduce_zero_label=False,
        pipeline=test_pipeline))

norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='{spec["backbone"]}',
        pretrained=r'{encoder_checkpoint}',
        style='pytorch'),
    decode_head=dict(
        type='SegMANDecoder',
        in_channels={spec["in_channels"]!r},
        in_index=[0, 1, 2, 3],
        channels={spec["channels"]},
        feat_proj_dim={spec["feat_proj_dim"]},
        dropout_ratio=0.1,
        num_classes={len(ALL_CLASSES)},
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(custom_keys=dict(
        pos_block=dict(decay_mult=0.0),
        norm=dict(decay_mult=0.0),
        head=dict(lr_mult=10.0),
    )))
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')
lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters={max_iters})
checkpoint_config = dict(by_epoch=False, interval={val_interval}, max_keep_ckpts=2)
evaluation = dict(interval={val_interval}, metric='mIoU', save_best='mIoU')
'''


def write_segman_config(
    segman_root: str | Path,
    variant: str,
    encoder_checkpoint: str | Path,
    export_root: str | Path | None = None,
    max_iters: int = 30000,
    val_interval: int = 1000,
    batch_size: int = 2,
    workers: int = 4,
) -> Path:
    segman_root = Path(segman_root)
    export_root = get_segman_export_root() if export_root is None else Path(export_root)
    config_path = (
        segman_root
        / "segmentation"
        / "local_configs"
        / "segman"
        / "ga2"
        / f"segman_{variant.lower()}_ga2.py"
    )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        build_segman_config_text(
            variant=variant,
            export_root=export_root,
            encoder_checkpoint=encoder_checkpoint,
            max_iters=max_iters,
            val_interval=val_interval,
            batch_size=batch_size,
            workers=workers,
        )
    )
    return config_path
