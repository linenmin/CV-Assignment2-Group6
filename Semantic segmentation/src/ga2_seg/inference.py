from pathlib import Path

import numpy as np
import torch
from mmseg.apis import inference_model, init_model

from .mmseg_runtime import load_config
from .submission import iter_test_images


def build_inference_config(config_path: str | Path | None = None):
    cfg = load_config(config_path)
    cfg = cfg.copy()
    inference_pipeline = [
        dict(type="LoadImageFromNDArray"),
        dict(type="PackSegInputs"),
    ]
    cfg.test_pipeline = inference_pipeline
    cfg.test_dataloader.dataset.pipeline = inference_pipeline
    return cfg


def resolve_inference_device(device: str | None = None) -> str:
    if device is not None:
        return device
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def load_segmentation_model(
    checkpoint_path: str | Path,
    config_path: str | Path | None = None,
    device: str | None = None,
):
    cfg = build_inference_config(config_path)
    return init_model(
        cfg,
        checkpoint=str(Path(checkpoint_path).resolve()),
        device=resolve_inference_device(device),
    )


def predict_segmentation_mask(model, image: np.ndarray) -> np.ndarray:
    result = inference_model(model, image)
    return result.pred_sem_seg.data.squeeze(0).detach().cpu().numpy().astype(np.uint8)


def save_test_predictions(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    config_path: str | Path | None = None,
    device: str | None = None,
    limit: int | None = None,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_segmentation_model(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device=device,
    )

    written_paths: list[Path] = []
    for index, (test_id, image) in enumerate(iter_test_images(limit=limit), start=1):
        prediction = predict_segmentation_mask(model, image)
        output_path = output_dir / f"test_{test_id}.npy"
        np.save(output_path, prediction)
        written_paths.append(output_path)

        if index % 50 == 0:
            print(f"predicted={index}")

    return written_paths
