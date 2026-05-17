from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .labels import CLASS_NAME_TO_LABEL_ID, PALETTE

IGNORE_LABEL = 255

CITYSCAPES_LABEL_TO_VOC = {
    24: CLASS_NAME_TO_LABEL_ID["person"],
    26: CLASS_NAME_TO_LABEL_ID["car"],
    28: CLASS_NAME_TO_LABEL_ID["bus"],
    31: CLASS_NAME_TO_LABEL_ID["train"],
    32: CLASS_NAME_TO_LABEL_ID["motorbike"],
    33: CLASS_NAME_TO_LABEL_ID["bicycle"],
}

OVERLAP_CLASS_NAMES = (
    "person",
    "car",
    "bus",
    "train",
    "motorbike",
    "bicycle",
)

# ``train`` appears in both label sets only by name. Visual inspection of
# ``outputs/figures/train_class_voc_vs_cityscapes/`` shows VOC trains are
# full-frame intercity / steam locomotives photographed as the subject,
# while Cityscapes "trains" are urban trams that appear small and
# incidentally in street scenes. These are essentially different visual
# concepts, so the primary external-generalization metric should exclude
# ``train`` and report only the genuinely transferable classes below.
TRANSFERABLE_CLASS_NAMES = (
    "person",
    "car",
    "bus",
    "motorbike",
    "bicycle",
)


@dataclass(frozen=True)
class CityscapesPair:
    image_path: Path
    label_path: Path
    sample_id: str


@dataclass(frozen=True)
class CityscapesRoots:
    left_img_root: Path
    gt_fine_root: Path
    split: str


def _resolve_named_root(root: Path, expected_name: str) -> Path:
    root = root.resolve()
    if root.name == expected_name:
        return root
    direct = root / expected_name
    if direct.exists():
        return direct.resolve()
    nested = root / f"{expected_name}_trainvaltest" / expected_name
    if nested.exists():
        return nested.resolve()
    return direct.resolve()


def resolve_cityscapes_roots(
    cityscapes_root: str | Path | None = None,
    left_img_root: str | Path | None = None,
    gt_fine_root: str | Path | None = None,
    split: str = "val",
) -> CityscapesRoots:
    if cityscapes_root is None and (left_img_root is None or gt_fine_root is None):
        raise ValueError("Provide --cityscapes-root or both --left-img-root and --gt-fine-root.")

    base = Path(cityscapes_root) if cityscapes_root is not None else None
    left_root = _resolve_named_root(Path(left_img_root) if left_img_root else base, "leftImg8bit")
    gt_root = _resolve_named_root(Path(gt_fine_root) if gt_fine_root else base, "gtFine")
    return CityscapesRoots(left_img_root=left_root, gt_fine_root=gt_root, split=split)


def collect_cityscapes_pairs(roots: CityscapesRoots, limit: int | None = None) -> list[CityscapesPair]:
    image_root = roots.left_img_root / roots.split
    label_root = roots.gt_fine_root / roots.split
    image_paths = sorted(image_root.glob("*/*_leftImg8bit.png"))

    if not image_paths:
        raise FileNotFoundError(f"No Cityscapes images found under {image_root}")
    if not label_root.exists():
        raise FileNotFoundError(
            f"Cityscapes ground-truth split not found: {label_root}. "
            "For validation metrics, extract gtFine_trainvaltest.zip so gtFine/val exists."
        )

    pairs: list[CityscapesPair] = []
    for image_path in image_paths:
        city = image_path.parent.name
        sample_id = image_path.name.replace("_leftImg8bit.png", "")
        label_path = label_root / city / f"{sample_id}_gtFine_labelIds.png"
        if label_path.exists():
            pairs.append(CityscapesPair(image_path=image_path, label_path=label_path, sample_id=sample_id))

    if not pairs:
        raise FileNotFoundError(
            f"No image/label pairs found for split={roots.split}. "
            f"Checked images under {image_root} and labels under {label_root}."
        )

    return pairs[:limit] if limit is not None else pairs


def map_cityscapes_label_ids(label_ids: np.ndarray) -> np.ndarray:
    mapped = np.full(label_ids.shape, IGNORE_LABEL, dtype=np.uint8)
    for cityscapes_id, voc_id in CITYSCAPES_LABEL_TO_VOC.items():
        mapped[label_ids == cityscapes_id] = voc_id
    return mapped


def resize_prediction_to_label(prediction: np.ndarray, label_shape: tuple[int, int]) -> np.ndarray:
    if prediction.shape == label_shape:
        return prediction.astype(np.uint8, copy=False)
    image = Image.fromarray(prediction.astype(np.uint8), mode="L")
    resized = image.resize((label_shape[1], label_shape[0]), resample=Image.Resampling.NEAREST)
    return np.asarray(resized, dtype=np.uint8)


def compute_overlap_intersections_unions(
    prediction: np.ndarray,
    target: np.ndarray,
    class_ids: tuple[int, ...],
) -> tuple[dict[int, int], dict[int, int], dict[int, int]]:
    valid = target != IGNORE_LABEL
    intersections: dict[int, int] = {}
    unions: dict[int, int] = {}
    target_pixels: dict[int, int] = {}

    for class_id in class_ids:
        pred_mask = (prediction == class_id) & valid
        target_mask = target == class_id
        intersections[class_id] = int(np.logical_and(pred_mask, target_mask).sum())
        unions[class_id] = int(np.logical_or(pred_mask, target_mask).sum())
        target_pixels[class_id] = int(target_mask.sum())

    return intersections, unions, target_pixels


def summarize_iou(
    intersections: dict[int, int],
    unions: dict[int, int],
    target_pixels: dict[int, int],
) -> list[dict[str, object]]:
    class_name_by_id = {CLASS_NAME_TO_LABEL_ID[name]: name for name in OVERLAP_CLASS_NAMES}
    rows: list[dict[str, object]] = []
    for class_id in sorted(class_name_by_id):
        union = unions[class_id]
        intersection = intersections[class_id]
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_name_by_id[class_id],
                "intersection": intersection,
                "union": union,
                "target_pixels": target_pixels[class_id],
                "iou": float(intersection / union) if union else float("nan"),
            }
        )
    return rows


def colorize_overlap_mask(mask: np.ndarray) -> np.ndarray:
    palette = np.asarray(PALETTE, dtype=np.uint8)
    output = np.zeros((*mask.shape, 3), dtype=np.uint8)
    valid_ids = [CLASS_NAME_TO_LABEL_ID[name] for name in OVERLAP_CLASS_NAMES]
    for class_id in valid_ids:
        output[mask == class_id] = palette[class_id]
    return output
