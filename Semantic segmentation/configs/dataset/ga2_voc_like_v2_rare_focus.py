dataset_type = "GA2SegDataset"
data_root = r"D:\BaiduNetdiskWorkspace\Leuven\8th\Computer Vision\assignment\Group2\kul-computer-vision-ga-2-2026"
project_root = r"D:\BaiduNetdiskWorkspace\Leuven\8th\Computer Vision\assignment\Group2\Semantic segmentation"
crop_size = (512, 512)
rare_focus_class_ids = [2, 9, 10, 16, 17]

train_pipeline = [
    dict(type="LoadNpyImageFromFile"),
    dict(type="LoadNpySegAnnotations"),
    dict(type="RandomResize", scale=(1024, 512), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(
        type="RareClassFocusedCrop",
        crop_size=crop_size,
        rare_class_ids=rare_focus_class_ids,
        rare_prob=0.5,
        cat_max_ratio=0.65,
        max_attempts=10,
    ),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]

test_pipeline = [
    dict(type="LoadNpyImageFromFile"),
    dict(type="LoadNpySegAnnotations"),
    dict(type="PackSegInputs"),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=project_root + r"\data\splits\train_rare_focus_v2.txt",
        data_prefix=dict(
            img_path=data_root + r"\train\img",
            seg_map_path=data_root + r"\train\seg",
        ),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=project_root + r"\data\splits\val.txt",
        data_prefix=dict(
            img_path=data_root + r"\train\img",
            seg_map_path=data_root + r"\train\seg",
        ),
        pipeline=test_pipeline,
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(type="IoUMetric", iou_metrics=["mIoU"])
test_evaluator = val_evaluator
