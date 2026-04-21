from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .runtime_compat import configure_windows_runtime


configure_windows_runtime()

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class RunArtifacts:
    work_dir: Path
    run_dir: Path
    scalars_path: Path
    log_path: Path
    analysis_dir: Path


def find_run_artifacts(work_dir: str | Path) -> RunArtifacts:
    work_dir = Path(work_dir).resolve()
    run_dirs = sorted(
        path
        for path in work_dir.iterdir()
        if path.is_dir() and (path / "vis_data" / "scalars.json").exists()
    )
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in {work_dir}")

    run_dir = run_dirs[-1]
    scalars_path = run_dir / "vis_data" / "scalars.json"
    if not scalars_path.exists():
        raise FileNotFoundError(f"Missing scalars file: {scalars_path}")

    log_candidates = sorted(run_dir.glob("*.log"))
    if not log_candidates:
        raise FileNotFoundError(f"Missing log file in {run_dir}")

    return RunArtifacts(
        work_dir=work_dir,
        run_dir=run_dir,
        scalars_path=scalars_path,
        log_path=log_candidates[-1],
        analysis_dir=work_dir / "analysis",
    )


def load_scalar_records(scalars_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    for line in Path(scalars_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))

    df = pd.DataFrame.from_records(records)
    train_df = df[df["loss"].notna()].copy() if "loss" in df else pd.DataFrame()
    val_df = df[df["mIoU"].notna()].copy() if "mIoU" in df else pd.DataFrame()

    if not train_df.empty:
        train_df = train_df.sort_values("step").reset_index(drop=True)
    if not val_df.empty:
        val_df = val_df.sort_values("step").reset_index(drop=True)

    return train_df, val_df


def lookup_kaggle_score(
    tracker_path: str | Path,
    submission_filename: str | None = None,
    checkpoint_name: str | None = None,
) -> dict[str, str] | None:
    path = Path(tracker_path)
    if not path.exists():
        return None

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| 20"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 7:
            continue

        record = {
            "date": cells[0],
            "submission_file": cells[1].strip("`"),
            "checkpoint": cells[2].strip("`"),
            "local_val_miou": cells[3].strip("`"),
            "kaggle_score": cells[4].strip("`"),
            "classification_placeholder": cells[5].strip("`"),
            "notes": cells[6],
        }
        if submission_filename and record["submission_file"].endswith(submission_filename):
            return record
        if checkpoint_name and record["checkpoint"].endswith(checkpoint_name):
            return record

    return None


def infer_best_checkpoint_name(work_dir: str | Path) -> str | None:
    candidates = sorted(Path(work_dir).glob("best_mIoU_iter_*.pth"))
    return candidates[-1].name if candidates else None


def build_training_summary(
    val_records: list[dict] | pd.DataFrame,
    submission_filename: str | None = None,
    checkpoint_name: str | None = None,
    kaggle_entry: dict[str, str] | None = None,
) -> dict[str, object]:
    val_df = pd.DataFrame(val_records).sort_values("step").reset_index(drop=True)
    if val_df.empty:
        raise ValueError("Validation records are required to build a summary.")

    best_row = val_df.loc[val_df["mIoU"].idxmax()]
    stop_row = val_df.iloc[-1]
    summary = {
        "best_step": int(best_row["step"]),
        "best_miou": float(best_row["mIoU"]),
        "best_macc": float(best_row["mAcc"]),
        "best_aacc": float(best_row["aAcc"]),
        "stop_step": int(stop_row["step"]),
        "last_miou": float(stop_row["mIoU"]),
        "submission_filename": submission_filename,
        "checkpoint_name": checkpoint_name,
        "kaggle_score": None,
        "classification_placeholder": None,
    }

    if kaggle_entry:
        summary["kaggle_score"] = kaggle_entry.get("kaggle_score")
        summary["classification_placeholder"] = kaggle_entry.get("classification_placeholder")

    return summary


def plot_training_curves(train_df: pd.DataFrame, val_df: pd.DataFrame, output_path: str | Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    axes[0].plot(train_df["step"], train_df["loss"], label="train loss", color="#d1495b")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(val_df["step"], val_df["mIoU"], marker="o", label="val mIoU", color="#00798c")
    axes[1].plot(val_df["step"], val_df["mAcc"], marker="o", label="val mAcc", color="#edae49")
    axes[1].plot(val_df["step"], val_df["aAcc"], marker="o", label="val aAcc", color="#30638e")
    axes[1].set_ylabel("Metric")
    axes[1].set_title("Validation Metrics")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(train_df["step"], train_df["lr"], label="learning rate", color="#003d5b")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("LR")
    axes[2].set_title("Learning Rate Schedule")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_runtime_diagnostics(train_df: pd.DataFrame, output_path: str | Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    axes[0].plot(train_df["step"], train_df["decode.acc_seg"], color="#2a9d8f")
    axes[0].set_ylabel("decode.acc_seg")
    axes[0].set_title("Segmentation Accuracy Proxy")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(train_df["step"], train_df["data_time"], color="#8d99ae")
    axes[1].set_ylabel("data_time")
    axes[1].set_title("Data Loading Time")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(train_df["step"], train_df["time"], color="#6d597a")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("iter time")
    axes[2].set_title("Iteration Time")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_validation_metrics_csv(val_df: pd.DataFrame, output_path: str | Path) -> None:
    columns = ["step", "mIoU", "mAcc", "aAcc", "data_time", "time"]
    export_df = val_df[[column for column in columns if column in val_df.columns]].copy()
    export_df.to_csv(output_path, index=False)


def write_summary_markdown(summary: dict[str, object], output_path: str | Path) -> None:
    lines = [
        "# Training Run Summary",
        "",
        f"- Best step: `{summary['best_step']}`",
        f"- Best val mIoU: `{summary['best_miou']:.2f}`",
        f"- Best val mAcc: `{summary['best_macc']:.2f}`",
        f"- Best val aAcc: `{summary['best_aacc']:.2f}`",
        f"- Stop step: `{summary['stop_step']}`",
        f"- Last val mIoU: `{summary['last_miou']:.2f}`",
    ]

    if summary.get("checkpoint_name"):
        lines.append(f"- Best checkpoint: `{summary['checkpoint_name']}`")
    if summary.get("submission_filename"):
        lines.append(f"- Submission file: `{summary['submission_filename']}`")
    if summary.get("kaggle_score"):
        lines.append(f"- Kaggle score: `{summary['kaggle_score']}`")
    if summary.get("classification_placeholder"):
        lines.append(
            f"- Classification placeholder: `{summary['classification_placeholder']}`"
        )

    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
