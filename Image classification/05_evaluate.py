"""Step 5: evaluate one configured classification experiment."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from shared import (  # noqa: E402
    DATA_DIR,
    EXPERIMENTS,
    LABELS,
    NUM_WORKERS,
    ExperimentConfig,
    describe_experiment,
    ensure_experiment_dirs,
    get_experiment_config,
    load_module,
)

_here = Path(__file__).parent
_ds = load_module("dataset", _here / "02_dataset.py")
_mdl = load_module("model", _here / "03_model.py")

VOCDataset = _ds.VOCDataset
load_train_df = _ds.load_train_df
get_val_transform = _ds.get_val_transform
MultiLabelClassifier = _mdl.MultiLabelClassifier


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        choices=sorted(EXPERIMENTS),
        default=None,
        help="Experiment config to evaluate.",
    )
    parser.add_argument(
        "--ckpt",
        choices=["best", "final", "last"],
        default="best",
        help="Checkpoint from the experiment folder.",
    )
    return parser.parse_args()


def resolve_checkpoint(config: ExperimentConfig, ckpt: str) -> Path:
    if ckpt == "final":
        return config.final_checkpoint
    if ckpt == "last":
        return config.last_checkpoint
    return config.best_checkpoint


def build_val_loader(data_dir: Path, config: ExperimentConfig, device: torch.device):
    df = load_train_df(data_dir)
    _, val_idx = train_test_split(
        range(len(df)),
        test_size=config.val_split,
        random_state=config.random_seed,
    )
    val_ds = VOCDataset(
        df.iloc[val_idx],
        data_dir,
        split="train",
        transform=get_val_transform(config.img_size, mode=config.transform_mode),
    )
    return DataLoader(
        val_ds,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )


def collect_preds(model, loader, device: torch.device):
    all_logits, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            logits = model(imgs.to(device)).cpu()
            all_logits.append(logits)
            all_labels.append(labels)

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs = 1 / (1 + np.exp(-logits))
    return probs, labels


def find_best_thresholds(probs, labels, thresholds=np.arange(0.1, 0.91, 0.05)):
    best_thresholds = []
    for class_idx in range(probs.shape[1]):
        best_t, best_f1 = 0.5, 0.0
        for threshold in thresholds:
            preds = (probs[:, class_idx] > threshold).astype(int)
            f1 = f1_score(labels[:, class_idx], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, threshold
        best_thresholds.append(best_t)
    return np.array(best_thresholds)


def main(
    config: ExperimentConfig | None = None,
    ckpt_path: Path | None = None,
    ckpt_name: str = "best",
):
    if config is None:
        args = parse_args()
        config = get_experiment_config(args.experiment)
        ckpt_name = args.ckpt

    ensure_experiment_dirs(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = Path(ckpt_path) if ckpt_path else resolve_checkpoint(config, ckpt_name)

    if not path.exists():
        print(f"Checkpoint not found: {path}")
        print("Run 04_train.py first.")
        return None

    print(f"Device: {device}")
    print(f"Experiment: {describe_experiment(config)}")
    print(f"Checkpoint: {path}")

    model = MultiLabelClassifier(
        backbone=config.backbone,
        num_classes=len(LABELS),
        pretrained=False,
    ).to(device)
    model.load_state_dict(torch.load(path, map_location=device))

    val_loader = build_val_loader(DATA_DIR, config, device)
    probs, labels = collect_preds(model, val_loader, device)

    rows = []
    print(f"\n{'Class':<15s}  {'AP':>6s}")
    print("-" * 25)
    for i, label in enumerate(LABELS):
        ap = average_precision_score(labels[:, i], probs[:, i])
        rows.append({"class": label, "ap": ap})
        print(f"{label:<15s}  {ap:6.4f}")

    ap_df = pd.DataFrame(rows)
    map_score = float(ap_df["ap"].mean())
    print("-" * 25)
    print(f"{'mAP':<15s}  {map_score:6.4f}")

    best_thresholds = find_best_thresholds(probs, labels)
    ap_df["best_threshold"] = best_thresholds
    ap_df.to_csv(config.ap_csv_path, index=False)
    np.save(config.thresholds_path, best_thresholds)

    summary = pd.DataFrame(
        [
            {
                "experiment": config.name,
                "checkpoint": path.name,
                "backbone": config.backbone,
                "img_size": config.img_size,
                "mAP": map_score,
            }
        ]
    )
    summary.to_csv(config.eval_summary_path, index=False)

    print("\nBest thresholds per class:")
    for label, threshold in zip(LABELS, best_thresholds):
        print(f"  {label:<15s}: {threshold:.2f}")
    print(f"\nThresholds saved to {config.thresholds_path}")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(LABELS, ap_df["ap"], color="steelblue")
    ax.axhline(map_score, color="red", linestyle="--", label=f"mAP={map_score:.3f}")
    ax.set_ylabel("Average Precision")
    ax.set_title(f"Per-class AP on validation set ({config.name})")
    ax.set_xticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(config.ap_plot_path, dpi=100)
    plt.close()
    print(f"AP chart saved to {config.ap_plot_path}")
    print(f"AP CSV saved to {config.ap_csv_path}")
    return map_score


if __name__ == "__main__":
    main()
