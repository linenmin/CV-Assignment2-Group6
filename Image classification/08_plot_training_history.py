"""Plot training-loss history from saved experiment CSV files."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from shared import CLASSIFICATION_OUTPUT_DIR, EXPERIMENTS, ExperimentConfig, get_experiment_config  # noqa: E402


def plot_training_history(
    history_df: pd.DataFrame,
    output_path: Path,
    experiment_name: str,
) -> Path | None:
    """Save a train/validation loss chart matching the Colab notebook style."""

    history_df = history_df.copy()
    if history_df.empty:
        print(f"No training history rows to plot for {experiment_name}.")
        return None

    required = {"stage", "train_loss", "val_loss"}
    missing = required.difference(history_df.columns)
    if missing:
        raise ValueError(f"{experiment_name} history is missing columns: {sorted(missing)}")

    history_df["step"] = np.arange(1, len(history_df) + 1)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(
        history_df["step"],
        history_df["train_loss"],
        marker="o",
        linewidth=1.8,
        label="train loss",
    )

    val_df = history_df.dropna(subset=["val_loss"])
    if not val_df.empty:
        ax.plot(
            val_df["step"],
            val_df["val_loss"],
            marker="s",
            linewidth=1.8,
            label="val loss",
        )

    stage_starts = history_df.groupby("stage", sort=False)["step"].min()
    ymax = float(np.nanmax(history_df[["train_loss", "val_loss"]].to_numpy()))
    for stage, start in stage_starts.items():
        ax.axvline(start, color="0.85", linewidth=0.8, zorder=0)
        ax.text(start, ymax, stage, va="top", ha="left", fontsize=9, color="0.35")

    ax.set_title(f"{experiment_name} training history")
    ax.set_xlabel("training step (epoch within each stage)")
    ax.set_ylabel("AsymmetricLoss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    print(f"Saved figure: {output_path}")
    return output_path


def plot_experiment(config: ExperimentConfig) -> Path | None:
    if not config.training_history_path.exists():
        print(f"Missing history CSV: {config.training_history_path}")
        return None
    history_df = pd.read_csv(config.training_history_path)
    return plot_training_history(history_df, config.training_history_plot_path, config.name)


def iter_history_dirs(root: Path):
    for history_path in sorted(root.glob("*/metrics/training_history.csv")):
        yield history_path.parent.parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        choices=sorted(EXPERIMENTS),
        default=None,
        help="Experiment config to plot. Omit with --all to scan output folders.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Plot every output folder that contains metrics/training_history.csv.",
    )
    return parser.parse_args()


def main(config: ExperimentConfig | None = None):
    if config is not None:
        return [plot_experiment(config)]

    args = parse_args()
    if args.all:
        outputs = []
        for experiment_dir in iter_history_dirs(CLASSIFICATION_OUTPUT_DIR):
            history_path = experiment_dir / "metrics" / "training_history.csv"
            output_path = experiment_dir / "figures" / "training_history.png"
            history_df = pd.read_csv(history_path)
            outputs.append(plot_training_history(history_df, output_path, experiment_dir.name))
        return outputs

    config = get_experiment_config(args.experiment)
    return [plot_experiment(config)]


if __name__ == "__main__":
    main()
