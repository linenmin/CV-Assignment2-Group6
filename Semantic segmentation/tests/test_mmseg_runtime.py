from ga2_seg.mmseg_runtime import build_dataset, load_config, resolve_config_path


def test_mmseg_runtime_resolves_default_experiment_config():
    assert resolve_config_path().name == "segnext_s_512x512_adamw_poly_v1.py"


def test_mmseg_train_dataset_builds_and_returns_first_sample():
    cfg = load_config()
    dataset = build_dataset(cfg, split="train")
    sample = dataset[0]
    assert len(dataset) == 637
    assert "inputs" in sample
    assert "data_samples" in sample


def test_mmseg_val_sample_keeps_label_shape_aligned_with_original_size():
    cfg = load_config()
    dataset = build_dataset(cfg, split="val")
    sample = dataset[0]
    assert tuple(sample["data_samples"].gt_sem_seg.data.shape[-2:]) == tuple(
        sample["data_samples"].metainfo["ori_shape"]
    )
