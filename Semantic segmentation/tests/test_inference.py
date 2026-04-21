from ga2_seg.inference import build_inference_config


def test_build_inference_config_uses_ndarray_test_pipeline():
    cfg = build_inference_config()
    dataset_pipeline = cfg.test_dataloader.dataset.pipeline
    assert dataset_pipeline[0]["type"] == "LoadImageFromNDArray"
    assert dataset_pipeline[-1]["type"] == "PackSegInputs"
    assert cfg.test_pipeline[0]["type"] == "LoadImageFromNDArray"
    assert cfg.test_pipeline[-1]["type"] == "PackSegInputs"
