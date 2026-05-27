"""Helpers for rendering Chapter 3 segmentation results.

The module loads the bundled CSV tables and produces the four
chapter-3 visualisations: the experiments-comparison styled table, the
pivot-attribution waterfall, the lab-vs-Kaggle scatter, the loss
ablation grouped bar chart, the V17 training curve, the V17 per-class
IoU bar chart, and the cross-task per-class comparison against the
classifier's per-class AP.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


def load_experiments_table(path: str | Path) -> pd.DataFrame:
    """Load the V1..V17 experiments table."""
    return pd.read_csv(path)


def style_experiments_table(df: pd.DataFrame, chosen_version: str = "V17") -> "pd.io.formats.style.Styler":
    def _highlight_row(row: pd.Series) -> list[str]:
        if row["version"] == chosen_version:
            return ["font-weight: bold; background-color: #e8f3ff"] * len(row)
        return [""] * len(row)

    styler = df.style.apply(_highlight_row, axis=1)
    styler = styler.format({
        "val_miou": "{:.2f}",
        "kaggle_seg_dice": "{:.4f}",
        "kaggle_joint_dice": "{:.4f}",
    }, na_rep="—")
    styler = styler.hide(axis="index")
    return styler


def plot_pivot_waterfall(df: pd.DataFrame, ax: plt.Axes | None = None, baseline_value: float | None = None) -> plt.Axes:
    """Waterfall chart attributing Kaggle seg-half Dice gains to pivots.

    The chart walks from V1 (baseline) to V17 (current best), drawing one
    bar per version with height equal to the version's delta over the
    previous one. Bars are coloured by the pivot family so the reader
    can see at a glance which pivot type moved the needle.
    """

    pivot_groups = {
        "baseline / loss tweak": ["baseline (MMSeg + ADE20K-init)", "rare-class oversampling",
                                  "region-rebalance auxiliary head", "hard-pixel sampler (OHEM)",
                                  "composite loss (CE + Dice)"],
        "model family swap": ["model family swap", "model family swap (CVPR 2025)",
                              "model family + DINOv3 pretrain"],
        "backbone scale-up": ["backbone scale-up"],
        "ensemble": ["hard-vote V6+V7+V8"],
        "training continuation": ["training continuation (lr=1e-5)",
                                  "low-LR continuation + early stop",
                                  "layer-wise LR + cosine + unfreeze-4"],
        "inference TTA": ["inference TTA (multi-scale 496/512/528)"],
        "training-data expansion": ["training-data expansion (712/37 split)"],
    }
    colours = {
        "baseline / loss tweak": "#c44e52",
        "model family swap": "#4c72b0",
        "backbone scale-up": "#55a868",
        "ensemble": "#8172b3",
        "training continuation": "#ccb974",
        "inference TTA": "#64b5cd",
        "training-data expansion": "#dd8452",
    }

    def _which_group(pivot: str) -> str:
        for name, keys in pivot_groups.items():
            if pivot in keys:
                return name
        return "other"

    if ax is None:
        _, ax = plt.subplots(figsize=(11.0, 5.5), dpi=130)

    df = df.copy().reset_index(drop=True)
    df["group"] = df["pivot"].apply(_which_group)
    # If the caller supplied an external baseline (e.g. V1's absolute Dice
    # after filtering V1 out), use it as the reference for the first row's
    # delta. Otherwise the first row keeps its raw value, matching the old
    # behaviour.
    diffs = df["kaggle_seg_dice"].diff()
    if baseline_value is not None and len(df) > 0:
        first_delta = float(df["kaggle_seg_dice"].iloc[0]) - float(baseline_value)
        diffs.iloc[0] = first_delta
    df["delta"] = diffs.fillna(df["kaggle_seg_dice"])

    bar_colours = [colours[g] for g in df["group"]]
    bars = ax.bar(df["version"], df["delta"], color=bar_colours)
    ax.axhline(0, color="#444444", linewidth=0.7)
    ax.set_ylabel("Kaggle segmentation-half Dice delta vs previous version", fontsize=11)
    base_str = f" (baseline V1 = {baseline_value:.3f})" if baseline_value is not None else ""
    ax.set_title(
        f"Pivot attribution: per-version delta in Kaggle seg-half Dice{base_str}",
        fontsize=12,
    )
    ax.tick_params(axis="x", labelsize=10)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    handles = [mpatches.Patch(color=col, label=name) for name, col in colours.items()]
    ax.legend(handles=handles, fontsize=9, loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=5, frameon=False, handlelength=1.4, columnspacing=1.1)
    return ax


def plot_lab_vs_kaggle_scatter(df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    """Scatter plot of val mIoU (x) vs Kaggle seg-half Dice (y) per version.

    Points are coloured by model family so the journey from SegNeXt-S
    (V1-V5) through SegFormer (V6-V9), SegMAN (V10), and EoMT-DINOv3
    (V11-V17) is visible without crowding text labels in the upper-right.
    A few landmark versions (V1, V10, V11, V17) are still labelled
    inline; the others are identified by colour against the legend.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8.5, 6.0), dpi=130)

    valid = df.dropna(subset=["val_miou", "kaggle_seg_dice"]).copy()
    val_x = valid["val_miou"] / 100.0
    kaggle_y = valid["kaggle_seg_dice"]

    family_colour = {
        "SegNeXt-S":           "#c44e52",
        "SegFormer-B2":        "#4c72b0",
        "SegFormer-B3":        "#4c72b0",
        "SegFormer-B5":        "#4c72b0",
        "SegFormer ensemble":  "#8172b3",
        "SegMAN-B":            "#55a868",
        "EoMT-DINOv3-L":       "#dd8452",
    }
    family_label_order = [
        "SegNeXt-S (V1-V5)",
        "SegFormer family (V6-V9)",
        "SegFormer ensemble (V9)",
        "SegMAN-B (V10)",
        "EoMT-DINOv3-L (V11-V17)",
    ]
    family_label_colour = {
        "SegNeXt-S (V1-V5)":         "#c44e52",
        "SegFormer family (V6-V9)":  "#4c72b0",
        "SegFormer ensemble (V9)":   "#8172b3",
        "SegMAN-B (V10)":            "#55a868",
        "EoMT-DINOv3-L (V11-V17)":   "#dd8452",
    }

    colours = [family_colour.get(fam, "#888888") for fam in valid["model_family"]]
    ax.scatter(val_x, kaggle_y, s=80, c=colours, edgecolor="white", linewidth=1.2, zorder=3)

    # Inline labels only for the journey landmarks (V1, V10, V11, V17).
    landmark_offsets = {
        "V1":  (8, 4),
        "V10": (8, -14),
        "V11": (-26, -16),
        "V17": (8, -14),
    }
    for _, row in valid.iterrows():
        version = row["version"]
        if version in landmark_offsets:
            ax.annotate(
                version,
                (row["val_miou"] / 100.0, row["kaggle_seg_dice"]),
                textcoords="offset points",
                xytext=landmark_offsets[version],
                fontsize=10, fontweight="bold",
            )

    lo = min(val_x.min(), kaggle_y.min()) - 0.02
    hi = max(val_x.max(), kaggle_y.max()) + 0.02
    ax.plot([lo, hi], [lo, hi], color="#888888", linestyle="--", linewidth=0.8)
    ax.text(hi - 0.015, hi - 0.025, "y = x", fontsize=10, color="#6b7280",
            horizontalalignment="right", verticalalignment="top", fontstyle="italic")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Internal validation mIoU (0-1 scale)", fontsize=11)
    ax.set_ylabel("Kaggle segmentation-half Dice", fontsize=11)
    ax.set_title("Lab vs Kaggle: does the local val track the hidden test?", fontsize=12)

    import matplotlib.lines as mlines
    handles = [
        mlines.Line2D([], [], marker="o", color=family_label_colour[lbl],
                       markersize=8, markeredgecolor="white", linewidth=0, label=lbl)
        for lbl in family_label_order
    ]
    ax.legend(handles=handles, fontsize=9, loc="lower right", frameon=True,
              edgecolor="#cbd5e1", facecolor="white", framealpha=0.95)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return ax


def load_loss_ablation(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def plot_loss_ablation(df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(9.0, 5.0), dpi=130)
    x = np.arange(len(df))
    width = 0.36
    val = df["val_miou"] / 100.0
    seg = df["kaggle_seg_dice"]
    bars_val = ax.bar(x - width / 2, val, width, color="#4c72b0", label="Validation mIoU (0-1)")
    bars_seg = ax.bar(x + width / 2, seg, width, color="#dd8452", label="Kaggle seg-half Dice")
    # Numerical value labels on every bar so the small height differences
    # (the headline finding) are unambiguous.
    for b in list(bars_val) + list(bars_seg):
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.008, f"{h:.3f}",
                ha="center", va="bottom", fontsize=10, color="#1f2937")
    ax.set_xticks(x)
    ax.set_xticklabels(df["version"] + "\n" + df["loss"], fontsize=10)
    # Extra headroom so the value labels do not collide with the legend.
    ax.set_ylim(0, max(val.max(), seg.max()) + 0.12)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(
        "Loss ablation on SegNeXt-S: validation mIoU rose, Kaggle Dice fell",
        fontsize=12,
    )
    ax.legend(fontsize=10, loc="upper right", frameon=False, bbox_to_anchor=(1.0, 1.06))
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return ax


def load_training_log(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def plot_training_curve(df: pd.DataFrame, best_step: int = 1000, ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8.5, 4.5), dpi=130)
    ax.plot(df["step"], df["val_miou"], marker="o", color="#4c72b0", linewidth=1.6)
    best_y = float(df.loc[df["step"] == best_step, "val_miou"].iloc[0])
    ax.axvline(best_step, color="#c44e52", linestyle="--", linewidth=0.8,
               label=f"best @ step {best_step} (val mIoU = {best_y:.4f})")
    ax.set_xlabel("Optimisation step", fontsize=11)
    ax.set_ylabel("Validation mIoU (37-image val)", fontsize=11)
    ax.set_title("V17 training curve: best at step 1000, early-stopped at 1800", fontsize=12)
    ax.legend(fontsize=10, loc="lower right", frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return ax


def load_per_class_iou(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def plot_per_class_iou(df: pd.DataFrame, ax: plt.Axes | None = None, weak_threshold: float = 0.5) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8.5, 6.5), dpi=130)
    df = df[df["class_name"] != "background"].dropna(subset=["iou"]).copy()
    df = df.sort_values("iou").reset_index(drop=True)
    colours = ["#c44e52" if v < weak_threshold else "#4c72b0" for v in df["iou"]]
    ax.barh(df["class_name"], df["iou"], color=colours)
    ax.axvline(weak_threshold, color="#888888", linestyle="--", linewidth=0.8)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Per-class IoU on validation", fontsize=11)
    ax.set_title(
        f"V17 per-class IoU (red bars < {weak_threshold:.1f})",
        fontsize=12,
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return ax


def load_cityscapes_transferable(path: str | Path) -> pd.DataFrame:
    """Load the Cityscapes 6-class overlap CSV."""
    return pd.read_csv(path)


def plot_cityscapes_transferable(df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    """Horizontal bar chart of Cityscapes overlap-class IoU on V17.

    The 5-class transferable scope (``person, car, bus, motorbike, bicycle``)
    is plotted in colour; ``train`` is included for transparency but plotted
    in light grey because we exclude it from the headline metric.
    """

    if ax is None:
        _, ax = plt.subplots(figsize=(8.5, 4.5), dpi=130)
    df = df.sort_values("iou").reset_index(drop=True)
    colours = ["#bbbbbb" if name == "train" else "#4c72b0" for name in df["class_name"]]
    labels = [
        f"{name}  (excluded)" if name == "train" else name
        for name in df["class_name"]
    ]
    ax.barh(labels, df["iou"], color=colours)
    ax.set_xlim(0, 1.0)
    ax.axvline(0.5, color="#888888", linestyle=":", linewidth=0.7)
    ax.set_xlabel("Per-class IoU on Cityscapes val (500 images, no Cityscapes training)", fontsize=11)
    transferable = df[df["class_name"] != "train"]
    mean_5 = float(transferable["iou"].mean())
    ax.set_title(
        f"Cross-domain evaluation, V17 on Cityscapes: 5-class transferable mIoU = {mean_5:.3f}",
        fontsize=12,
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return ax


def plot_cross_task_per_class(seg_df: pd.DataFrame, cls_df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    """Side-by-side bars per class: classification AP vs segmentation IoU.

    Sort by segmentation IoU ascending so the weak segmentation classes
    are at the top. Classes missing from either dataset get NaN bars.
    """

    if ax is None:
        _, ax = plt.subplots(figsize=(9.5, 7.5), dpi=130)
    seg_df = seg_df[seg_df["class_name"] != "background"].rename(columns={"iou": "seg_iou"})
    cls_df = cls_df.rename(columns={"class": "class_name", "ap": "cls_ap"})
    merged = seg_df.merge(cls_df, on="class_name", how="outer")
    merged = merged.sort_values("seg_iou", ascending=True, na_position="last").reset_index(drop=True)

    y = np.arange(len(merged))
    height = 0.4
    ax.barh(y - height / 2, merged["seg_iou"], height, color="#4c72b0",
            label="Segmentation IoU (V17)")
    ax.barh(y + height / 2, merged["cls_ap"], height, color="#dd8452",
            label="Classification AP (ConvNeXt-Small)")
    ax.set_yticks(y)
    ax.set_yticklabels(merged["class_name"], fontsize=10)
    ax.set_xlim(0, 1.05)
    ax.axvline(0.5, color="#888888", linestyle=":", linewidth=0.7)
    ax.set_xlabel("Per-class score (0-1 scale)", fontsize=11)
    ax.set_title(
        "Cross-task per-class: shared and task-specific failure modes",
        fontsize=12,
    )
    ax.legend(fontsize=10, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.16), frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return ax
