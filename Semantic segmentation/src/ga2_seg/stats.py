from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .labels import CLASS_NAMES
from .paths import get_dataset_root


def load_train_labels_df() -> pd.DataFrame:
    dataset_root = get_dataset_root()
    train_df = pd.read_csv(dataset_root / "train" / "train_set.csv", index_col="Id")
    return train_df.loc[:, list(CLASS_NAMES)]


def compute_class_presence_counts(train_df: pd.DataFrame | None = None) -> dict[str, int]:
    if train_df is None:
        train_df = load_train_labels_df()
    return {label: int(train_df[label].sum()) for label in CLASS_NAMES}


def compute_image_shape_counts() -> dict[str, int]:
    dataset_root = get_dataset_root()
    shape_counts: Counter[tuple[int, ...]] = Counter()
    for img_path in sorted((dataset_root / "train" / "img").glob("*.npy")):
        shape_counts[np.load(img_path, mmap_mode="r").shape] += 1
    return {str(shape): count for shape, count in sorted(shape_counts.items(), key=lambda item: (-item[1], item[0]))}


def compute_pixel_label_counts() -> dict[str, int]:
    dataset_root = get_dataset_root()
    pixel_counts: Counter[int] = Counter()
    for seg_path in sorted((dataset_root / "train" / "seg").glob("*.npy")):
        values, counts = np.unique(np.load(seg_path), return_counts=True)
        for value, count in zip(values, counts):
            pixel_counts[int(value)] += int(count)
    return {str(label_id): pixel_counts[label_id] for label_id in sorted(pixel_counts)}


def build_stats_summary() -> dict:
    train_df = load_train_labels_df()
    dataset_root = get_dataset_root()
    return {
        "dataset_root": str(dataset_root),
        "train_examples": int(len(train_df)),
        "num_classes": len(CLASS_NAMES),
        "class_names": list(CLASS_NAMES),
        "class_presence_counts": compute_class_presence_counts(train_df),
        "image_shape_counts": compute_image_shape_counts(),
        "pixel_label_counts": compute_pixel_label_counts(),
    }

