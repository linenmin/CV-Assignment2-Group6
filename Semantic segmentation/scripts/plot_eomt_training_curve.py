"""Plot and summarize EoMT training curves from a CSV log."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.log)

    best_index = int(df["val_miou"].idxmax())
    best_row = df.loc[best_index]
    final_row = df.iloc[-1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=160)
    axes[0].plot(df["step"], df["val_miou"], marker="o", color="#19535f")
    axes[0].scatter([best_row["step"]], [best_row["val_miou"]], color="#d95d39", zorder=3)
    axes[0].set_title("Validation mIoU")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("mIoU")
    axes[0].grid(alpha=0.25)

    axes[1].plot(df["step"], df["loss"], marker="o", color="#6b4e71")
    axes[1].set_title("Training Loss")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Loss")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    figure_path = args.output_dir / "training_curve.png"
    fig.savefig(figure_path)
    plt.close(fig)

    summary = [
        "# EoMT Training Curve Summary",
        "",
        f"- Log: `{args.log}`",
        f"- Best validation mIoU: `{best_row['val_miou']:.4f}` at step `{int(best_row['step'])}`",
        f"- Final validation mIoU: `{final_row['val_miou']:.4f}` at step `{int(final_row['step'])}`",
        f"- Final loss: `{final_row['loss']:.4f}`",
        f"- Figure: `{figure_path}`",
        "",
        "## Interpretation",
        "",
    ]
    if int(best_row["step"]) == int(final_row["step"]):
        summary.append("- Validation mIoU is still highest at the final checkpoint, so this run likely has remaining training headroom.")
    else:
        summary.append("- Validation mIoU peaked before the final checkpoint, so the next run should use stronger early stopping or a lower learning rate.")
    summary.append("- Loss is noisy because each logged point is a single minibatch loss, not an epoch average.")

    summary_path = args.output_dir / "training_curve_summary.md"
    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"figure={figure_path}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
