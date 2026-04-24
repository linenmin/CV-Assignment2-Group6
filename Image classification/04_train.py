"""
Step 4: Two-stage ResNet-50 fine-tuning for PASCAL VOC multi-label classification.

Run from project root:
    python "Image classification/04_train.py"

Output:
    output/checkpoints/best_model.pth   (best val-loss checkpoint)
    output/checkpoints/last_model.pth   (final epoch)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from shared import LABELS, DATA_DIR, OUTPUT_DIR, NUM_WORKERS, load_module, AsymmetricLoss  # noqa: E402

_here = Path(__file__).parent
_ds   = load_module("dataset", _here / "02_dataset.py")
_mdl  = load_module("model",   _here / "03_model.py")

VOCDataset            = _ds.VOCDataset
load_train_df         = _ds.load_train_df
get_train_transform   = _ds.get_train_transform
get_val_transform     = _ds.get_val_transform
MultiLabelClassifier  = _mdl.MultiLabelClassifier

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
BACKBONE   = "efficientnet_b3"  # or "resnet50"
IMG_SIZE   = 320                # larger input preserves small-object detail
BATCH_SIZE = 16                 # reduced from 32 to fit 320×320 in GPU memory

STAGE1_EPOCHS = 5
STAGE1_LR     = 1e-3

STAGE2_EPOCHS = 20
STAGE2_LR     = 1e-4

# Stage 3: retrain on ALL 750 samples using best weights
# Gives the final model that 06_predict.py will use
STAGE3_EPOCHS = 5
STAGE3_LR     = 5e-5

WEIGHT_DECAY  = 1e-4
VAL_SPLIT     = 0.2
RANDOM_SEED   = 42

CKPT_DIR = OUTPUT_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def build_loaders(data_dir: Path):
    df = load_train_df(data_dir)

    # Simple random split — sufficient for 750 samples
    train_idx, val_idx = train_test_split(
        range(len(df)), test_size=VAL_SPLIT, random_state=RANDOM_SEED
    )

    train_ds = VOCDataset(df.iloc[train_idx], data_dir, split="train",
                          transform=get_train_transform(IMG_SIZE))
    val_ds   = VOCDataset(df.iloc[val_idx],   data_dir, split="train",
                          transform=get_val_transform(IMG_SIZE))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # Positive-class weights to handle label imbalance
    labels_array = df[LABELS].values.astype(float)
    pos_counts   = labels_array.sum(axis=0)
    neg_counts   = len(df) - pos_counts
    pos_weight   = torch.FloatTensor(neg_counts / (pos_counts + 1e-6))

    return train_loader, val_loader, pos_weight


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------
def run_epoch(model, loader, criterion, optimizer, train: bool):
    model.train(train)
    total_loss = 0.0
    with torch.set_grad_enabled(train):
        for imgs, labels in tqdm(loader, leave=False,
                                 desc="train" if train else "val  "):
            imgs   = imgs.to(device)
            labels = labels.to(device)

            logits = model(imgs)
            loss   = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(imgs)

    return total_loss / len(loader.dataset)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_stage(model, train_loader, val_loader, criterion,
                lr, epochs, stage_name, best_val_loss):
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, train=True)
        val_loss   = run_epoch(model, val_loader,   criterion, optimizer, train=False)
        scheduler.step()

        flag = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CKPT_DIR / "best_model.pth")
            flag = "  ← best"

        print(f"[{stage_name}] Epoch {epoch:3d}/{epochs} | "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}{flag}")

    return best_val_loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    train_loader, val_loader, pos_weight = build_loaders(DATA_DIR)

    model = MultiLabelClassifier(backbone=BACKBONE, num_classes=len(LABELS),
                                  pretrained=True).to(device)
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05)

    best_val_loss = float("inf")

    # ---- Stage 1: frozen backbone, train head only ----
    print(f"\n=== Stage 1: Training classification head (backbone={BACKBONE}, frozen) ===")
    model.freeze_backbone()
    print(f"Trainable params: {model.trainable_params():,}")
    best_val_loss = train_stage(
        model, train_loader, val_loader, criterion,
        lr=STAGE1_LR, epochs=STAGE1_EPOCHS,
        stage_name="S1", best_val_loss=best_val_loss,
    )

    # ---- Stage 2: full fine-tune ----
    print("\n=== Stage 2: Full fine-tuning (backbone unfrozen) ===")
    model.unfreeze_backbone()
    print(f"Trainable params: {model.trainable_params():,}")
    best_val_loss = train_stage(
        model, train_loader, val_loader, criterion,
        lr=STAGE2_LR, epochs=STAGE2_EPOCHS,
        stage_name="S2", best_val_loss=best_val_loss,
    )

    # ---- Stage 3: retrain on ALL data with best weights ----
    # Removes the 20% val holdout penalty — gives the model 25% more training signal.
    # Uses a very low LR to avoid overwriting the well-trained features.
    print("\n=== Stage 3: Retrain on full dataset (all 750 samples) ===")
    model.load_state_dict(torch.load(CKPT_DIR / "best_model.pth", map_location=device))
    full_df = load_train_df(DATA_DIR)
    full_ds = VOCDataset(full_df, DATA_DIR, split="train",
                         transform=get_train_transform(IMG_SIZE))
    full_loader = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=NUM_WORKERS, pin_memory=True)
    optimizer3  = torch.optim.AdamW(model.parameters(), lr=STAGE3_LR,
                                     weight_decay=WEIGHT_DECAY)
    scheduler3  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer3,
                                                               T_max=STAGE3_EPOCHS)
    for epoch in range(1, STAGE3_EPOCHS + 1):
        loss = run_epoch(model, full_loader, criterion, optimizer3, train=True)
        scheduler3.step()
        print(f"[S3] Epoch {epoch:3d}/{STAGE3_EPOCHS} | full_train_loss={loss:.4f}")

    torch.save(model.state_dict(), CKPT_DIR / "final_model.pth")
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints: best_model.pth (val-tuned)  final_model.pth (full-data)")


if __name__ == "__main__":
    main()
