"""Data-overview helpers for Chapter 1 of the report notebook.

The functions here load the competition data (mounted at
``/kaggle/input/kul-computer-vision-ga-2-2026/`` on Kaggle, or any local
copy with the same layout) and produce the two visual surfaces used in
Chapter 1: the per-class image-count bar chart and a 2x2 grid of sample
images with their segmentation masks. A small offline helper to
pre-render the grid into a PNG is included so the notebook can simply
load the pre-built figure at runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap


CLASS_NAMES: tuple[str, ...] = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)


# 21-colour palette (background + 20 VOC classes). Matches the official
# PASCAL VOC colour map used in most published figures.
_VOC_PALETTE = np.asarray(
    [
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
        [64, 0, 128],
        [192, 0, 128],
        [64, 128, 128],
        [192, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 192, 0],
        [128, 192, 0],
        [0, 64, 128],
    ],
    dtype=np.uint8,
)


def compute_class_distribution(data_dir: str | Path) -> pd.DataFrame:
    """Count how many training images contain each VOC class.

    Returns
    -------
    pd.DataFrame
        Two columns, ``class_name`` and ``image_count``, sorted by count
        ascending so a horizontal bar chart reads naturally from rare
        classes at the bottom to common classes at the top.
    """

    train_csv = Path(data_dir) / "train" / "train_set.csv"
    df = pd.read_csv(train_csv, index_col="Id")
    counts = df[list(CLASS_NAMES)].sum().astype(int).rename("image_count")
    out = counts.reset_index().rename(columns={"index": "class_name"})
    return out.sort_values("image_count").reset_index(drop=True)


def plot_class_distribution(counts: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    """Render a horizontal bar chart of per-class image counts."""

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    ax.barh(counts["class_name"], counts["image_count"], color="#4c72b0")
    ax.set_xlabel("Number of training images containing the class")
    ax.set_ylabel("")
    ax.set_title("Per-class image count (749 training images, 20 VOC classes)")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return ax


def _colourise_mask(mask: np.ndarray) -> np.ndarray:
    palette = _VOC_PALETTE
    indices = np.clip(mask.astype(np.int64), 0, len(palette) - 1)
    return palette[indices]


def _load_train_pair(data_dir: Path, sample_id: int) -> tuple[np.ndarray, np.ndarray]:
    image = np.load(data_dir / "train" / "img" / f"train_{sample_id}.npy")
    mask = np.load(data_dir / "train" / "seg" / f"train_{sample_id}.npy")
    return image.astype(np.uint8), mask.astype(np.uint8)


def _classes_for_sample(data_dir: Path, sample_id: int) -> list[str]:
    """Return the list of VOC classes present in a training image (image-level labels)."""
    csv = pd.read_csv(data_dir / "train" / "train_set.csv", index_col="Id")
    row = csv.loc[sample_id, list(CLASS_NAMES)]
    return [name for name in CLASS_NAMES if row[name] == 1]


def _resize_to_square(arr: np.ndarray, side: int, interp_nearest: bool) -> np.ndarray:
    """Resize an image or a label map to ``side x side`` for uniform panels."""
    from PIL import Image as PILImage

    if arr.ndim == 3:
        pil = PILImage.fromarray(arr.astype(np.uint8))
        resample = PILImage.Resampling.BILINEAR
    else:
        pil = PILImage.fromarray(arr.astype(np.uint8), mode="L")
        resample = PILImage.Resampling.NEAREST if interp_nearest else PILImage.Resampling.BILINEAR
    out = pil.resize((side, side), resample)
    return np.asarray(out, dtype=arr.dtype)


def render_sample_grid(
    data_dir: str | Path,
    sample_ids: Iterable[int],
    fig: plt.Figure | None = None,
    panel_side: int = 256,
) -> plt.Figure:
    """Render a 2 x 4 grid of (image, mask) panels directly into a figure.

    Every panel is resized to ``panel_side x panel_side`` so all subplots
    share the same shape; this keeps captions clear of neighbouring
    panels regardless of the original image aspect ratio. The file
    prefix ``train_`` in the source data only marks the training split,
    so panel captions list the actual VOC classes present.
    """

    data_dir = Path(data_dir)
    sample_ids = list(sample_ids)
    if len(sample_ids) != 4:
        raise ValueError(f"Expected exactly 4 sample ids, got {len(sample_ids)}.")

    if fig is None:
        fig = plt.figure(figsize=(11.0, 6.0), dpi=130)
    axes = fig.subplots(2, 4)

    for col, sample_id in enumerate(sample_ids):
        image, mask = _load_train_pair(data_dir, sample_id)
        image_sq = _resize_to_square(image, panel_side, interp_nearest=False)
        mask_sq = _resize_to_square(mask, panel_side, interp_nearest=True)
        coloured = _colourise_mask(mask_sq)
        classes = _classes_for_sample(data_dir, sample_id)
        caption = ", ".join(classes) if classes else "background only"

        axes[0, col].imshow(image_sq)
        axes[0, col].set_title(f"Image #{sample_id}\n{caption}", fontsize=12)
        axes[1, col].imshow(coloured)
        axes[1, col].set_title("ground-truth mask", fontsize=11)
        for row in range(2):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            for spine in axes[row, col].spines.values():
                spine.set_edgecolor("#cccccc")

    fig.suptitle(
        "Four training examples: image (top) and per-pixel class mask (bottom)",
        fontsize=14,
        y=1.03,
    )
    fig.tight_layout(h_pad=2.0, w_pad=0.8)
    return fig
