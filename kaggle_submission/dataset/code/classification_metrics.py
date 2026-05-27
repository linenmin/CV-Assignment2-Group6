"""Helpers for rendering Chapter 2 classification results.

Loads the bundled experiment-comparison CSV and per-class AP CSV, and
provides a horizontal bar plot of per-class AP sorted ascending so the
weakest classes are immediately visible. Everything is intentionally
lightweight; the notebook calls these helpers and prints / plots the
results inline.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_experiments_table(path: str | Path) -> pd.DataFrame:
    """Load the multi-backbone experiment comparison table."""
    return pd.read_csv(path)


def load_per_class_ap(path: str | Path) -> pd.DataFrame:
    """Load the per-class AP + optimal-threshold table for the final model."""
    df = pd.read_csv(path)
    df = df.rename(columns={"class": "class_name", "ap": "AP", "best_threshold": "threshold"})
    df = df.sort_values("AP").reset_index(drop=True)
    return df


def style_experiments_table(df: pd.DataFrame, chosen_experiment: str) -> "pd.io.formats.style.Styler":
    """Return a pandas Styler that bolds the chosen-experiment row."""
    def _highlight_row(row: pd.Series) -> list[str]:
        if row["experiment_name"] == chosen_experiment:
            return ["font-weight: bold; background-color: #e8f3ff"] * len(row)
        return [""] * len(row)

    styler = df.style.apply(_highlight_row, axis=1)
    styler = styler.format({"val_mAP": "{:.3f}", "kaggle_dice": "{:.3f}"}, na_rep="—")
    styler = styler.hide(axis="index")
    return styler


def load_loss_ablation(bce_path: str | Path, asl_path: str | Path) -> pd.DataFrame:
    """Merge per-class AP from BCE and AsymmetricLoss runs of the same recipe."""
    bce = pd.read_csv(bce_path).rename(columns={"ap": "BCE"})
    asl = pd.read_csv(asl_path).rename(columns={"ap": "AsymmetricLoss"})
    df = bce.merge(asl, on="class")
    df["delta"] = df["AsymmetricLoss"] - df["BCE"]
    return df.sort_values("delta", ascending=False).reset_index(drop=True)


def plot_loss_ablation(df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    """Horizontal diverging bar chart of per-class AP delta (ASL minus BCE)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8.5, 6.0), dpi=130)
    ordered = df.sort_values("delta")
    colours = ["#c44e52" if d < 0 else "#4c72b0" for d in ordered["delta"]]
    ax.barh(ordered["class"], ordered["delta"], color=colours)
    ax.axvline(0, color="#444444", linewidth=0.8)
    ax.set_xlabel("Per-class AP delta (AsymmetricLoss minus BCE)", fontsize=11)
    ax.set_title(
        "Loss ablation, ConvNeXt-Small 320, same recipe and seed (25 epochs: 5 Stage 1 + 20 Stage 2)",
        fontsize=11,
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return ax


def plot_per_class_ap(df: pd.DataFrame, ax: plt.Axes | None = None, weak_threshold: float = 0.80) -> plt.Axes:
    """Horizontal bar chart of per-class AP. Bars below ``weak_threshold`` are highlighted."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8.5, 6.0), dpi=130)

    colours = ["#c44e52" if ap < weak_threshold else "#4c72b0" for ap in df["AP"]]
    ax.barh(df["class_name"], df["AP"], color=colours)
    ax.set_xlabel("Average Precision (AP) on validation", fontsize=11)
    ax.set_xlim(0, 1.02)
    ax.axvline(weak_threshold, color="#888888", linestyle="--", linewidth=0.8)
    ax.set_title(
        "Per-class AP, final ConvNeXt-Small model (red bars < {:.2f})".format(weak_threshold),
        fontsize=12,
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return ax
