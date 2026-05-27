"""Helpers for Chapter 4 (adversarial attacks).

Two styled tables (classifier attack, segmenter attack), one side-by-
side image display helper for the trainable-vs-PGD comparison, and a
single-image display helper for the standalone attack grids.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_classification_attack_results(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def style_classification_attacks(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    styler = df.style.format({
        "epsilon_normalized": "{:.3f}",
        "pixel_linf_mean": "{:.5f}",
        "clean_target_prob": "{:.4f}",
        "adv_target_prob": "{:.4f}",
        "attack_success_rate": "{:.2%}",
    })
    styler = styler.hide(axis="index")
    return styler


def load_segmentation_attack_results(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def style_segmentation_attacks(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    styler = df.style.format({
        "epsilon_pixel": "{:.1f}",
        "pixel_linf_mean": "{:.4f}",
        "clean_miou": "{:.2f}",
        "adv_miou": "{:.2f}",
        "miou_drop": "{:.2f}",
        "prediction_change_rate": "{:.2%}",
        "target_region_match_rate": "{:.2%}",
    }, na_rep="—")
    styler = styler.hide(axis="index")
    return styler


def show_attack_image(path: str | Path, title: str | None = None,
                       column_labels: tuple[str, ...] | None = None,
                       ax: plt.Axes | None = None) -> plt.Figure | plt.Axes:
    """Display the pre-rendered attack-example PNG as a single image.

    The source PNG already contains the 5x5 grid layout, per-cell
    captions and per-cell metrics. We just show it. A matplotlib
    title is optional. Earlier attempts to re-slice the grid created
    off-by-one alignment errors; the source is treated as one image
    here on purpose. The `column_labels` argument is accepted for
    backward compatibility with previous notebook calls and is
    ignored — the source PNG already carries column captions inline.
    """
    del column_labels  # source already labels its own columns

    img = plt.imread(str(path))
    if ax is None:
        fig_w = 13.0
        fig_h = img.shape[0] * fig_w / img.shape[1]
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
    else:
        fig = ax.figure
    ax.imshow(img, interpolation="lanczos")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=15, fontweight="600", pad=10)
    return fig


def show_two_attacks_side_by_side(left_path: str | Path, right_path: str | Path,
                                  left_title: str, right_title: str) -> plt.Figure:
    """Stack the trainable-vs-PGD comparison vertically.

    The source PNGs each contain a 5x5 grid with column captions and
    per-cell metrics already drawn. Their top ~40 px carries an
    overlapping figure title that we crop away; everything below is
    shown as-is. A matplotlib title is rendered above each panel.
    """
    top_img = plt.imread(str(left_path))[40:, :, :]
    bot_img = plt.imread(str(right_path))[40:, :, :]

    fig_w = 13.0
    h_top = top_img.shape[0] * fig_w / top_img.shape[1]
    h_bot = bot_img.shape[0] * fig_w / bot_img.shape[1]
    fig, axes = plt.subplots(
        2, 1,
        figsize=(fig_w, h_top + h_bot + 2.0),
        dpi=200,
        gridspec_kw={"height_ratios": [h_top, h_bot]},
    )
    for ax, img, title in ((axes[0], top_img, left_title),
                            (axes[1], bot_img, right_title)):
        ax.imshow(img, interpolation="lanczos")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, fontsize=15, fontweight="600", pad=12)
    fig.tight_layout(h_pad=2.0)
    return fig
