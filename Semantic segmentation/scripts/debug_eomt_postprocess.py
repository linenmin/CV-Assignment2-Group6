"""Compare EoMT validation metrics under square and original-size post-processing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, EomtDinov3ForUniversalSegmentation


NUM_CLASSES = 21


def mean_iou(preds: list[np.ndarray], targets: list[np.ndarray], include_background: bool) -> float:
    class_ids = range(NUM_CLASSES) if include_background else range(1, NUM_CLASSES)
    values: list[float] = []
    for class_id in class_ids:
        intersection = 0
        union = 0
        for pred, target in zip(preds, targets):
            pred_mask = pred == class_id
            target_mask = target == class_id
            intersection += int(np.logical_and(pred_mask, target_mask).sum())
            union += int(np.logical_or(pred_mask, target_mask).sum())
        if union > 0:
            values.append(intersection / union)
    return float(np.mean(values)) if values else 0.0


def main() -> None:
    checkpoint = Path("outputs/checkpoints/eomt_dinov3_v11_unfreeze2/best")
    processor = AutoImageProcessor.from_pretrained(checkpoint)
    model = EomtDinov3ForUniversalSegmentation.from_pretrained(checkpoint).eval().to("cuda")

    stems = [
        line.strip()
        for line in Path("data/segman_ga2/validation.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    image_dir = Path("data/segman_ga2/images/validation")
    mask_dir = Path("data/segman_ga2/annotations/validation")

    preds_square: list[np.ndarray] = []
    targets_square: list[np.ndarray] = []
    preds_hf_original: list[np.ndarray] = []
    preds_nearest_original: list[np.ndarray] = []
    preds_manual_square_original: list[np.ndarray] = []
    targets_original: list[np.ndarray] = []

    with torch.inference_mode():
        for stem in stems:
            image = Image.open(image_dir / f"{stem}.png").convert("RGB")
            target_original = np.asarray(Image.open(mask_dir / f"{stem}.png"), dtype=np.int64)
            target_square = np.asarray(
                Image.open(mask_dir / f"{stem}.png").resize((512, 512), Image.Resampling.NEAREST),
                dtype=np.int64,
            )

            inputs = processor(images=image, return_tensors="pt").to("cuda")
            outputs = model(**inputs)
            square_image = image.resize((512, 512), Image.Resampling.BILINEAR)
            square_inputs = processor(images=square_image, return_tensors="pt").to("cuda")
            square_outputs = model(**square_inputs)

            pred_square = processor.post_process_semantic_segmentation(
                outputs,
                target_sizes=[(512, 512)],
            )[0].detach().cpu().numpy().astype(np.int64)
            pred_hf_original = processor.post_process_semantic_segmentation(
                outputs,
                target_sizes=[image.size[::-1]],
            )[0].detach().cpu().numpy().astype(np.int64)
            pred_nearest_original = np.asarray(
                Image.fromarray(pred_square.astype(np.uint8)).resize(image.size, Image.Resampling.NEAREST),
                dtype=np.int64,
            )
            pred_manual_square = processor.post_process_semantic_segmentation(
                square_outputs,
                target_sizes=[(512, 512)],
            )[0].detach().cpu().numpy().astype(np.int64)
            pred_manual_square_original = np.asarray(
                Image.fromarray(pred_manual_square.astype(np.uint8)).resize(image.size, Image.Resampling.NEAREST),
                dtype=np.int64,
            )

            preds_square.append(pred_square)
            targets_square.append(target_square)
            preds_hf_original.append(pred_hf_original)
            preds_nearest_original.append(pred_nearest_original)
            preds_manual_square_original.append(pred_manual_square_original)
            targets_original.append(target_original)

    for name, preds, targets in [
        ("square_vs_square", preds_square, targets_square),
        ("hf_original_vs_original", preds_hf_original, targets_original),
        ("nearest_original_vs_original", preds_nearest_original, targets_original),
        ("manual_square_nearest_original_vs_original", preds_manual_square_original, targets_original),
    ]:
        print(name)
        print("  miou_with_background", mean_iou(preds, targets, include_background=True))
        print("  miou_foreground_only", mean_iou(preds, targets, include_background=False))


if __name__ == "__main__":
    main()
