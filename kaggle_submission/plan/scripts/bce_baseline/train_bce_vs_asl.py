"""BCE vs AsymmetricLoss ablation on ConvNeXt-Small 320.

Reproduces the classification team's Stage 1 + Stage 2 training recipe twice,
once with AsymmetricLoss and once with plain BCEWithLogitsLoss, on the same
80/20 split (seed=42), then evaluates per-class Average Precision on the
held-out validation set. The two AP vectors are written to a CSV that the
notebook loads as the ablation table for the Loss-Ablation Pairing Rule.

This script intentionally skips Stage 3 (full-data retrain) because Stage 3
folds the validation images back into training and would make per-class AP
on the same split a self-evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))

from shared import (  # noqa: E402
    DATA_DIR,
    LABELS,
    NUM_WORKERS,
    AsymmetricLoss,
    ExperimentConfig,
    load_module,
)

# Resolve the dataset root robustly so the script works from any CWD.
PROJECT_ROOT = HERE.parents[3]
DATA_DIR = PROJECT_ROOT / "kul-computer-vision-ga-2-2026"

_ds = load_module("dataset", HERE / "02_dataset.py")
_mdl = load_module("model", HERE / "03_model.py")
VOCDataset = _ds.VOCDataset
load_train_df = _ds.load_train_df
get_train_transform = _ds.get_train_transform
get_val_transform = _ds.get_val_transform
MultiLabelClassifier = _mdl.MultiLabelClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "kaggle_submission" / "plan" / "scripts" / "bce_baseline" / "results",
    )
    parser.add_argument(
        "--loss",
        choices=("bce", "asymmetric", "both"),
        default="both",
    )
    parser.add_argument("--stage1-epochs", type=int, default=5)
    parser.add_argument("--stage2-epochs", type=int, default=20)
    parser.add_argument("--stage1-lr", type=float, default=1e-3)
    parser.add_argument("--stage2-lr", type=float, default=1e-4)
    parser.add_argument("--img-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_criterion(loss_name: str) -> nn.Module:
    if loss_name == "asymmetric":
        return AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05)
    if loss_name == "bce":
        return nn.BCEWithLogitsLoss()
    raise ValueError(f"Unknown loss: {loss_name}")


def make_loaders(img_size: int, batch_size: int, seed: int) -> tuple[DataLoader, DataLoader, pd.DataFrame]:
    df = load_train_df(DATA_DIR)
    train_idx, val_idx = train_test_split(range(len(df)), test_size=0.2, random_state=seed)
    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    train_ds = VOCDataset(train_df, DATA_DIR, split="train",
                          transform=get_train_transform(img_size, mode="resize"))
    val_ds = VOCDataset(val_df, DATA_DIR, split="train",
                        transform=get_val_transform(img_size, mode="resize"))
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=pin)
    return train_loader, val_loader, val_df


def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> float:
    model.train(train)
    total = 0.0
    with torch.set_grad_enabled(train):
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += float(loss.item()) * imgs.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def compute_val_metrics(model, val_loader, device) -> tuple[float, dict]:
    model.eval()
    all_probs, all_targets = [], []
    for imgs, labels in val_loader:
        imgs = imgs.to(device)
        probs = torch.sigmoid(model(imgs)).cpu().numpy()
        all_probs.append(probs)
        all_targets.append(labels.numpy())
    probs = np.concatenate(all_probs)
    targets = np.concatenate(all_targets)
    per_class = {}
    for i, name in enumerate(LABELS):
        if targets[:, i].sum() == 0:
            per_class[name] = float("nan")
        else:
            per_class[name] = float(average_precision_score(targets[:, i], probs[:, i]))
    mean_ap = float(np.nanmean(list(per_class.values())))
    return mean_ap, per_class


def train_one_recipe(loss_name: str, args, out_dir: Path) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== Loss: {loss_name} | device: {device} ===", flush=True)

    train_loader, val_loader, _ = make_loaders(args.img_size, args.batch_size, args.seed)

    model = MultiLabelClassifier(backbone="convnext_small", num_classes=len(LABELS), pretrained=True).to(device)
    criterion = build_criterion(loss_name)

    # Stage 1: freeze backbone, train head.
    backbone_module = next(iter(model.children()))
    for p in backbone_module.parameters():
        p.requires_grad = False
    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(head_params, lr=args.stage1_lr, weight_decay=1e-4)
    print("Stage 1 (head only)", flush=True)
    for epoch in range(args.stage1_epochs):
        t0 = time.time()
        tr = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        print(f"  ep{epoch+1}/{args.stage1_epochs} train_loss={tr:.4f} val_loss={va:.4f} ({time.time()-t0:.1f}s)", flush=True)

    # Stage 2: unfreeze and fine-tune.
    for p in model.parameters():
        p.requires_grad = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.stage2_lr, weight_decay=1e-4)
    best_mean_ap = -1.0
    best_per_class: dict = {}
    print("Stage 2 (full network)", flush=True)
    for epoch in range(args.stage2_epochs):
        t0 = time.time()
        tr = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        mean_ap, per_class = compute_val_metrics(model, val_loader, device)
        if mean_ap > best_mean_ap:
            best_mean_ap = mean_ap
            best_per_class = per_class
        print(f"  ep{epoch+1}/{args.stage2_epochs} train_loss={tr:.4f} val_mAP={mean_ap:.4f} best={best_mean_ap:.4f} ({time.time()-t0:.1f}s)", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"per_class_ap_{loss_name}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "ap"])
        for cls in LABELS:
            w.writerow([cls, best_per_class.get(cls, float("nan"))])
    print(f"Saved {csv_path}", flush=True)
    return {"loss": loss_name, "best_val_mAP": best_mean_ap, "per_class_ap": best_per_class}


def main() -> None:
    args = parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    losses = ["bce", "asymmetric"] if args.loss == "both" else [args.loss]
    for loss in losses:
        results[loss] = train_one_recipe(loss, args, out_dir)

    summary_path = out_dir / "ablation_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSummary: {summary_path}", flush=True)
    for loss, info in results.items():
        print(f"  {loss}: best_val_mAP = {info['best_val_mAP']:.4f}", flush=True)


if __name__ == "__main__":
    main()
