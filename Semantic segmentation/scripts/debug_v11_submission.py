"""Debug V11 EoMT submission quality and validation metric consistency."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoImageProcessor, EomtDinov3ForUniversalSegmentation


NUM_CLASSES = 21


def compute_iou(pred: np.ndarray, target: np.ndarray, include_background: bool) -> tuple[float, dict[int, float]]:
    class_range = range(NUM_CLASSES) if include_background else range(1, NUM_CLASSES)
    per_class: dict[int, float] = {}
    for class_id in class_range:
        pred_mask = pred == class_id
        target_mask = target == class_id
        union = np.logical_or(pred_mask, target_mask).sum()
        if union == 0:
            continue
        inter = np.logical_and(pred_mask, target_mask).sum()
        per_class[class_id] = float(inter / union)
    return float(np.mean(list(per_class.values()))) if per_class else 0.0, per_class


def validate_original_size(checkpoint: Path, limit: int | None) -> None:
    processor = AutoImageProcessor.from_pretrained(checkpoint)
    model = EomtDinov3ForUniversalSegmentation.from_pretrained(checkpoint).eval().to("cuda")
    image_dir = Path("data/segman_ga2/images/validation")
    mask_dir = Path("data/segman_ga2/annotations/validation")
    stems = [
        line.strip()
        for line in Path("data/segman_ga2/validation.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if limit is not None:
        stems = stems[:limit]

    intersections_bg = np.zeros(NUM_CLASSES, dtype=np.float64)
    unions_bg = np.zeros(NUM_CLASSES, dtype=np.float64)
    intersections_fg = np.zeros(NUM_CLASSES, dtype=np.float64)
    unions_fg = np.zeros(NUM_CLASSES, dtype=np.float64)
    label_counter: Counter[int] = Counter()

    with torch.inference_mode():
        for stem in stems:
            image = Image.open(image_dir / f"{stem}.png").convert("RGB")
            target = np.asarray(Image.open(mask_dir / f"{stem}.png"), dtype=np.int64)
            inputs = processor(images=image, return_tensors="pt").to("cuda")
            outputs = model(**inputs)
            pred = processor.post_process_semantic_segmentation(
                outputs,
                target_sizes=[image.size[::-1]],
            )[0].detach().cpu().numpy().astype(np.int64)
            label_counter.update(np.unique(pred).tolist())
            for class_id in range(NUM_CLASSES):
                pred_mask = pred == class_id
                target_mask = target == class_id
                union = np.logical_or(pred_mask, target_mask).sum()
                if union > 0:
                    intersections_bg[class_id] += np.logical_and(pred_mask, target_mask).sum()
                    unions_bg[class_id] += union
                if class_id > 0 and union > 0:
                    intersections_fg[class_id] += np.logical_and(pred_mask, target_mask).sum()
                    unions_fg[class_id] += union

    bg_ious = {
        class_id: float(intersections_bg[class_id] / unions_bg[class_id])
        for class_id in range(NUM_CLASSES)
        if unions_bg[class_id] > 0
    }
    fg_ious = {
        class_id: float(intersections_fg[class_id] / unions_fg[class_id])
        for class_id in range(1, NUM_CLASSES)
        if unions_fg[class_id] > 0
    }
    print("original_val_samples", len(stems))
    print("original_val_miou_with_background", float(np.mean(list(bg_ious.values()))))
    print("original_val_miou_foreground_only", float(np.mean(list(fg_ious.values()))))
    print("original_val_pred_labels", sorted(label_counter.items()))
    print("original_val_per_class_iou", fg_ious)


def summarize_predictions(prediction_dir: Path) -> None:
    paths = sorted(prediction_dir.glob("test_*.npy"), key=lambda path: int(path.stem.split("_")[1]))
    label_counter: Counter[int] = Counter()
    shape_counter: Counter[tuple[int, int]] = Counter()
    for path in paths:
        pred = np.load(path)
        shape_counter[tuple(pred.shape)] += 1
        label_counter.update(np.unique(pred).astype(int).tolist())
    print("prediction_count", len(paths))
    print("prediction_shapes", sorted(shape_counter.items(), key=lambda item: item[0])[:20])
    print("prediction_labels", sorted(label_counter.items()))


def summarize_csv(path: Path) -> None:
    df = pd.read_csv(path, keep_default_na=False)
    print("csv", path)
    print("rows", len(df))
    print("classification_rows", int(df["Id"].str.endswith("_classification").sum()))
    print("segmentation_rows", int(df["Id"].str.endswith("_segmentation").sum()))
    print("empty_predicted", int(df["Predicted"].eq("").sum()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/checkpoints/eomt_dinov3_v11_unfreeze2/best"))
    parser.add_argument("--prediction-dir", type=Path, default=Path("outputs/predictions/eomt_dinov3_v11_unfreeze2"))
    parser.add_argument("--csv", type=Path, default=Path("outputs/submissions/submission_exp_v11_eomt_dinov3_unfreeze2_with_convnext_small_320.csv"))
    parser.add_argument("--limit-val", type=int, default=None)
    args = parser.parse_args()

    summarize_csv(args.csv)
    summarize_predictions(args.prediction_dir)
    validate_original_size(args.checkpoint, args.limit_val)


if __name__ == "__main__":
    main()
