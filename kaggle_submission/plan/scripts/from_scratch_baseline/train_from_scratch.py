"""ConvNeXt-Tiny from-scratch baseline on PASCAL VOC classification.

Reproduces the same 80/20 split and AsymmetricLoss recipe used by the
production ConvNeXt-Small fine-tune, but with ConvNeXt-Tiny initialised
from random weights (pretrained=False). Trains for 25 epochs end-to-end
(no stage-1 head warm-up since there are no pretrained features to thaw)
and records validation mAP after each epoch.

The output CSV is consumed by Chapter 2 of the final notebook as the
measured evidence that supports the design choice of fine-tuning rather
than training from scratch on 749 images.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
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
BCE_DIR = HERE.parent / "bce_baseline"
sys.path.insert(0, str(BCE_DIR))

from shared import (  # noqa: E402
    LABELS,
    NUM_WORKERS,
    AsymmetricLoss,
    load_module,
)

PROJECT_ROOT = HERE.parents[3]
DATA_DIR = PROJECT_ROOT / "kul-computer-vision-ga-2-2026"

_ds = load_module("dataset", BCE_DIR / "02_dataset.py")
_mdl = load_module("model", BCE_DIR / "03_model.py")
VOCDataset = _ds.VOCDataset
load_train_df = _ds.load_train_df
get_train_transform = _ds.get_train_transform
get_val_transform = _ds.get_val_transform
MultiLabelClassifier = _mdl.MultiLabelClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=HERE / "results")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_loaders(img_size: int, batch_size: int, seed: int):
    df = load_train_df(DATA_DIR)
    train_idx, val_idx = train_test_split(
        range(len(df)), test_size=0.2, random_state=seed
    )
    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    train_ds = VOCDataset(
        train_df, DATA_DIR, split="train",
        transform=get_train_transform(img_size, mode="resize"),
    )
    val_ds = VOCDataset(
        val_df, DATA_DIR, split="train",
        transform=get_val_transform(img_size, mode="resize"),
    )
    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=pin,
    )
    return train_loader, val_loader, train_df, val_df


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
def compute_val_metrics(model, val_loader, device):
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
            per_class[name] = float(
                average_precision_score(targets[:, i], probs[:, i])
            )
    mean_ap = float(np.nanmean(list(per_class.values())))
    return mean_ap, per_class


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    train_loader, val_loader, train_df, val_df = make_loaders(
        args.img_size, args.batch_size, args.seed
    )
    print(f"Train size: {len(train_df)} | Val size: {len(val_df)}", flush=True)

    # Random init: pretrained=False.
    model = MultiLabelClassifier(
        backbone="convnext_tiny", num_classes=len(LABELS), pretrained=False
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"ConvNeXt-Tiny from scratch | total params: {total_params:,}", flush=True)

    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    history = []
    best_mean_ap = -1.0
    best_per_class: dict = {}
    best_epoch = -1
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )
        mean_ap, per_class = compute_val_metrics(model, val_loader, device)
        scheduler.step()
        is_best = mean_ap > best_mean_ap
        if is_best:
            best_mean_ap = mean_ap
            best_per_class = per_class
            best_epoch = epoch
        elapsed = time.time() - t0
        print(
            f"ep{epoch:2d}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_mAP={mean_ap:.4f} best={best_mean_ap:.4f}@{best_epoch} "
            f"({elapsed:.1f}s)",
            flush=True,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mAP": mean_ap,
                "is_best": int(is_best),
                "elapsed_sec": elapsed,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

    history_path = args.out_dir / "training_history_from_scratch.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    print(f"Saved: {history_path}", flush=True)

    per_class_path = args.out_dir / "per_class_ap_from_scratch.csv"
    with per_class_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "ap"])
        for cls in LABELS:
            w.writerow([cls, best_per_class.get(cls, float("nan"))])
    print(f"Saved: {per_class_path}", flush=True)

    summary = {
        "backbone": "convnext_tiny",
        "pretrained": False,
        "epochs": args.epochs,
        "lr": args.lr,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "loss": "asymmetric",
        "best_val_mAP": best_mean_ap,
        "best_epoch": best_epoch,
        "fine_tuned_reference_val_mAP": 0.900,
        "fine_tuned_reference_model": "convnext_small_320 ImageNet-1K init",
        "gap_to_finetune": 0.900 - best_mean_ap,
        "total_params": total_params,
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary: {summary_path}", flush=True)
    print(f"  best_val_mAP = {best_mean_ap:.4f} (epoch {best_epoch})", flush=True)
    print(f"  gap vs fine-tuned ConvNeXt-Small = {0.900 - best_mean_ap:.4f}", flush=True)


if __name__ == "__main__":
    main()
