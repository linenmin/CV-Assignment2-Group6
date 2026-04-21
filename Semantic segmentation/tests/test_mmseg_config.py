from mmengine.config import Config


def test_mmseg_experiment_config_loads_successfully():
    config_path = (
        r"D:\BaiduNetdiskWorkspace\Leuven\8th\Computer Vision\assignment\Group2\Semantic segmentation"
        r"\configs\experiments\segnext_s_512x512_adamw_poly_v1.py"
    )
    cfg = Config.fromfile(config_path)
    assert cfg.model.backbone.type == "MSCAN"
    assert cfg.model.decode_head.num_classes == 21
    assert "ade20k" in cfg.load_from.lower()


def test_dataset_config_uses_custom_npy_transforms():
    config_path = (
        r"D:\BaiduNetdiskWorkspace\Leuven\8th\Computer Vision\assignment\Group2\Semantic segmentation"
        r"\configs\experiments\segnext_s_512x512_adamw_poly_v1.py"
    )
    cfg = Config.fromfile(config_path)
    first_train_transform = cfg.train_dataloader.dataset.pipeline[0]["type"]
    second_train_transform = cfg.train_dataloader.dataset.pipeline[1]["type"]
    assert first_train_transform == "LoadNpyImageFromFile"
    assert second_train_transform == "LoadNpySegAnnotations"


def test_runtime_config_enables_delayed_early_stopping():
    config_path = (
        r"D:\BaiduNetdiskWorkspace\Leuven\8th\Computer Vision\assignment\Group2\Semantic segmentation"
        r"\configs\experiments\segnext_s_512x512_adamw_poly_v1.py"
    )
    cfg = Config.fromfile(config_path)
    early_stop_hook = cfg.custom_hooks[0]
    assert early_stop_hook["type"] == "DelayedEarlyStoppingHook"
    assert early_stop_hook["monitor"] == "mIoU"
    assert early_stop_hook["begin"] == 5
    assert early_stop_hook["patience"] == 6


def test_v2_rare_focus_config_uses_custom_split_and_crop():
    config_path = (
        r"D:\BaiduNetdiskWorkspace\Leuven\8th\Computer Vision\assignment\Group2\Semantic segmentation"
        r"\configs\experiments\segnext_s_512x512_adamw_poly_v2_rare_focus.py"
    )
    cfg = Config.fromfile(config_path)
    crop_transform = cfg.train_dataloader.dataset.pipeline[3]
    assert cfg.train_dataloader.dataset.ann_file.endswith(r"train_rare_focus_v2.txt")
    assert crop_transform["type"] == "RareClassFocusedCrop"
    assert crop_transform["cat_max_ratio"] == 0.65
