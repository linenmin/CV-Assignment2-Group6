"""Step 9: 5-fold cross-validation for per-class threshold calibration.

Trains S1+S2 on each fold (no S3), collects OOF TTA probabilities, then
calibrates per-class thresholds on the full 750-sample pool.  The resulting
best_thresholds_kfold.npy can be fed to 06_predict.py via --thresholds-path.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from shared import (  # noqa: E402
    DATA_DIR,
    EXPERIMENTS,
    LABELS,
    NUM_WORKERS,
    ExperimentConfig,
    build_criterion,
    describe_experiment,
    get_experiment_config,
    load_module,
)

_here = Path(__file__).parent
_ds = load_module("dataset", _here / "02_dataset.py")
_mdl = load_module("model", _here / "03_model.py")

VOCDataset = _ds.VOCDataset
load_train_df = _ds.load_train_df
get_train_transform = _ds.get_train_transform
get_val_transform = _ds.get_val_transform
MultiLabelClassifier = _mdl.MultiLabelClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="5-fold CV for OOF threshold calibration.")
    parser.add_argument(
        "--experiment",
        choices=sorted(EXPERIMENTS),
        default="convnext_small_320",
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=None,
        help="Specific fold indices to train (0-indexed). Default: all folds.",
    )
    parser.add_argument(
        "--mode",
        choices=["train-folds", "aggregate", "all"],
        default="all",
        help="train-folds: train fold models only. aggregate: calibrate thresholds only. all: both.",
    )
    parser.add_argument(
        "--grid-step",
        type=float,
        default=0.01,
        help="Threshold search grid step (default 0.01 = 91 candidates from 0.05 to 0.95).",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    model, loader, criterion, optimizer, train: bool, device: torch.device
) -> float:
    model.train(train)
    total_loss = 0.0
    with torch.set_grad_enabled(train):
        for imgs, labels in tqdm(loader, leave=False, desc="train" if train else "val  "):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(imgs)
    return total_loss / len(loader.dataset)


def train_fold(
    config: ExperimentConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    fold_k: int,
    kfold_dir: Path,
    device: torch.device,
) -> np.ndarray:
    """Train S1+S2 on one fold. Returns OOF TTA probabilities (N_val × 20)."""
    fold_dir = kfold_dir / f"fold{fold_k}"
    ckpt_dir = fold_dir / "checkpoints"
    metrics_dir = fold_dir / "metrics"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / "best_model.pth"

    set_seed(config.random_seed + fold_k)
    pin = device.type == "cuda"

    train_ds = VOCDataset(
        train_df, DATA_DIR, split="train",
        transform=get_train_transform(config.img_size, mode=config.transform_mode),
    )
    val_ds = VOCDataset(
        val_df, DATA_DIR, split="train",
        transform=get_val_transform(config.img_size, mode=config.transform_mode),
    )
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.eval_batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=pin,
    )

    model = MultiLabelClassifier(
        backbone=config.backbone, num_classes=len(LABELS), pretrained=True,
    ).to(device)
    criterion = build_criterion(config)
    history = []
    best_val_loss = float("inf")

    stages = [
        ("S1", config.stage1_lr, config.stage1_epochs, True),
        ("S2", config.stage2_lr, config.stage2_epochs, False),
    ]
    for stage, lr, epochs, freeze in stages:
        if freeze:
            model.freeze_backbone()
            print(f"  [Fold {fold_k}][S1] head-only  trainable={model.trainable_params():,}")
        else:
            model.unfreeze_backbone()
            print(f"  [Fold {fold_k}][S2] full       trainable={model.trainable_params():,}")

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        no_improve = 0

        for epoch in range(1, epochs + 1):
            train_loss = run_epoch(model, train_loader, criterion, optimizer, True, device)
            val_loss = run_epoch(model, val_loader, criterion, optimizer, False, device)
            scheduler.step()

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_ckpt)
                no_improve = 0
            else:
                no_improve += 1

            history.append({
                "fold": fold_k, "stage": stage, "epoch": epoch,
                "train_loss": train_loss, "val_loss": val_loss,
                "lr": lr, "is_best": is_best,
            })
            flag = "  <- best" if is_best else ""
            print(
                f"  [Fold {fold_k}][{stage}] Epoch {epoch:2d}/{epochs}  "
                f"train={train_loss:.4f}  val={val_loss:.4f}{flag}"
            )

            if (
                stage == "S2"
                and config.early_stop_patience
                and no_improve >= config.early_stop_patience
            ):
                print(f"  [Fold {fold_k}][S2] Early stopping after {no_improve} epochs.")
                break

    pd.DataFrame(history).to_csv(metrics_dir / "training_history.csv", index=False)

    # Reload best checkpoint, collect OOF probabilities with TTA
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    model.eval()
    all_probs = []
    with torch.no_grad():
        for imgs, _ in val_loader:
            imgs = imgs.to(device)
            p = (torch.sigmoid(model(imgs)) + torch.sigmoid(model(imgs.flip(-1)))) / 2
            all_probs.append(p.cpu().numpy())

    oof_probs = np.vstack(all_probs)
    print(f"  [Fold {fold_k}] OOF shape: {oof_probs.shape}  best_val_loss={best_val_loss:.4f}")
    return oof_probs


def aggregate_oof(
    config: ExperimentConfig,
    kfold_dir: Path,
    full_df: pd.DataFrame,
    fold_splits: list,
    n_folds: int,
    grid_step: float,
) -> float:
    """Load OOF probs from all fold metrics dirs, calibrate thresholds, save outputs."""
    print("\n=== Aggregating OOF probabilities ===")

    oof_probs_all = np.zeros((len(full_df), len(LABELS)))
    oof_labels_all = full_df[LABELS].values.astype(int)
    fold_thresholds = []

    for fold_k, (_, val_idx) in enumerate(fold_splits):
        oof_csv = kfold_dir / f"fold{fold_k}" / "metrics" / "oof_probabilities.csv"
        if not oof_csv.exists():
            print(f"  WARNING: {oof_csv} not found — skipping fold {fold_k}")
            continue
        fold_probs = pd.read_csv(oof_csv, index_col=0).values
        oof_probs_all[val_idx] = fold_probs

        fold_labels = oof_labels_all[val_idx]
        t_grid = np.arange(0.05, 0.96, grid_step)
        fold_t = []
        for c in range(len(LABELS)):
            best_t, best_f1 = 0.5, 0.0
            for t in t_grid:
                preds = (fold_probs[:, c] > t).astype(int)
                f1 = f1_score(fold_labels[:, c], preds, zero_division=0)
                if f1 > best_f1:
                    best_f1, best_t = f1, t
            fold_t.append(best_t)
        fold_t_arr = np.array(fold_t)
        fold_thresholds.append(fold_t_arr)
        np.save(kfold_dir / f"fold{fold_k}" / "metrics" / "thresholds.npy", fold_t_arr)
        print(f"  Fold {fold_k}: per-fold thresholds saved.")

    # Pooled OOF threshold search — use all 750 samples together (primary)
    t_grid = np.arange(0.05, 0.96, grid_step)
    ap_rows = []
    pooled_thresholds = []
    for c, label in enumerate(LABELS):
        best_t, best_f1 = 0.5, 0.0
        for t in t_grid:
            preds = (oof_probs_all[:, c] > t).astype(int)
            f1 = f1_score(oof_labels_all[:, c], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        pooled_thresholds.append(best_t)
        ap = average_precision_score(oof_labels_all[:, c], oof_probs_all[:, c])
        ap_rows.append({"class": label, "ap": ap, "best_threshold": best_t, "best_f1": best_f1})

    pooled_thresholds = np.array(pooled_thresholds)
    np.save(kfold_dir / "best_thresholds_kfold.npy", pooled_thresholds)

    # Average of per-fold thresholds (backup)
    if fold_thresholds:
        avg_thresholds = np.stack(fold_thresholds).mean(axis=0)
        np.save(kfold_dir / "best_thresholds_kfold_avg.npy", avg_thresholds)

    ap_df = pd.DataFrame(ap_rows)
    map_score = float(ap_df["ap"].mean())
    ap_df.to_csv(kfold_dir / "ap_per_class_kfold.csv", index=False)

    # Save aggregated OOF probabilities
    oof_df = pd.DataFrame(oof_probs_all, columns=LABELS, index=full_df.index)
    oof_df.index.name = "Id"
    oof_df.to_csv(kfold_dir / "oof_probabilities_all.csv")

    pd.DataFrame([{
        "experiment": config.name,
        "n_folds": n_folds,
        "oof_mAP": map_score,
    }]).to_csv(kfold_dir / "kfold_summary.csv", index=False)

    print(f"\n{'Class':<15s}  {'OOF AP':>6s}  {'Threshold':>9s}  {'OOF F1':>6s}")
    print("-" * 45)
    for _, row in ap_df.iterrows():
        print(
            f"{row['class']:<15s}  {row['ap']:6.4f}  {row['best_threshold']:9.3f}  {row['best_f1']:6.4f}"
        )
    print("-" * 45)
    print(f"{'OOF mAP':<15s}  {map_score:6.4f}")
    print(f"\nPooled thresholds  → {kfold_dir / 'best_thresholds_kfold.npy'}")
    if fold_thresholds:
        print(f"Averaged thresholds → {kfold_dir / 'best_thresholds_kfold_avg.npy'}")
    return map_score


def main(config: ExperimentConfig | None = None, args=None):
    if args is None:
        args = parse_args()
    if config is None:
        config = get_experiment_config(args.experiment)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kfold_dir = config.output_dir / "kfold"
    kfold_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Experiment: {describe_experiment(config)}")
    print(f"KFold output: {kfold_dir}")
    print(f"Mode: {args.mode} | n_folds: {args.n_folds} | grid_step: {args.grid_step}")

    full_df = load_train_df(DATA_DIR)
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    fold_splits = list(kf.split(full_df))

    folds_to_run = args.folds if args.folds is not None else list(range(args.n_folds))

    if args.mode in ("train-folds", "all"):
        print(f"\n=== Training {len(folds_to_run)}/{args.n_folds} fold(s) ===")
        for fold_k in folds_to_run:
            train_idx, val_idx = fold_splits[fold_k]
            print(f"\n{'='*60}")
            print(f"Fold {fold_k + 1}/{args.n_folds}:  {len(train_idx)} train / {len(val_idx)} val")
            print("=" * 60)
            oof_probs = train_fold(
                config,
                full_df.iloc[train_idx],
                full_df.iloc[val_idx],
                fold_k,
                kfold_dir,
                device,
            )
            metrics_dir = kfold_dir / f"fold{fold_k}" / "metrics"
            oof_df = pd.DataFrame(oof_probs, columns=LABELS, index=full_df.iloc[val_idx].index)
            oof_df.index.name = "Id"
            oof_df.to_csv(metrics_dir / "oof_probabilities.csv")
            print(f"  [Fold {fold_k}] OOF probabilities saved to {metrics_dir / 'oof_probabilities.csv'}")

    if args.mode in ("aggregate", "all"):
        aggregate_oof(config, kfold_dir, full_df, fold_splits, args.n_folds, args.grid_step)


if __name__ == "__main__":
    main()
