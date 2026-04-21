"""
Step 5: Evaluate the trained model on the validation split.

Run from project root:
    python "Image classification/05_evaluate.py"

Prints per-class AP and overall mAP.
Saves output/eval_confusion.png (per-class threshold sweep).
"""

import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, f1_score
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from shared import LABELS, DATA_DIR, OUTPUT_DIR, NUM_WORKERS, load_module  # noqa: E402

_here = Path(__file__).parent
_ds   = load_module("dataset", _here / "02_dataset.py")
_mdl  = load_module("model",   _here / "03_model.py")

VOCDataset         = _ds.VOCDataset
load_train_df      = _ds.load_train_df
get_val_transform  = _ds.get_val_transform
ResNet50Classifier = _mdl.ResNet50Classifier

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
VAL_SPLIT   = 0.2
RANDOM_SEED = 42
CKPT_PATH   = OUTPUT_DIR / "checkpoints" / "best_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Build validation loader (same split as training)
# ---------------------------------------------------------------------------
def build_val_loader(data_dir):
    df = load_train_df(data_dir)
    _, val_idx = train_test_split(
        range(len(df)), test_size=VAL_SPLIT, random_state=RANDOM_SEED
    )
    val_ds = VOCDataset(df.iloc[val_idx], data_dir, split="train",
                        transform=get_val_transform())
    return DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=NUM_WORKERS, pin_memory=True)


# ---------------------------------------------------------------------------
# Collect all predictions and ground-truth labels
# ---------------------------------------------------------------------------
def collect_preds(model, loader):
    all_logits, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            logits = model(imgs.to(device)).cpu()
            all_logits.append(logits)
            all_labels.append(labels)
    logits = torch.cat(all_logits).numpy()   # (N, 20)
    labels = torch.cat(all_labels).numpy()   # (N, 20)
    probs  = 1 / (1 + np.exp(-logits))       # sigmoid
    return probs, labels


# ---------------------------------------------------------------------------
# Find best per-class threshold (maximises F1 on val set)
# ---------------------------------------------------------------------------
def find_best_thresholds(probs, labels, thresholds=np.arange(0.1, 0.91, 0.05)):
    best_thresholds = []
    for c in range(probs.shape[1]):
        best_t, best_f1 = 0.5, 0.0
        for t in thresholds:
            preds = (probs[:, c] > t).astype(int)
            f1 = f1_score(labels[:, c], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        best_thresholds.append(best_t)
    return np.array(best_thresholds)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(ckpt_path=None):
    path = Path(ckpt_path) if ckpt_path else CKPT_PATH
    if not path.exists():
        print(f"Checkpoint not found: {path}")
        print("Run 04_train.py first.")
        return

    model = ResNet50Classifier(num_classes=len(LABELS), pretrained=False).to(device)
    model.load_state_dict(torch.load(path, map_location=device))

    val_loader = build_val_loader(DATA_DIR)
    probs, labels = collect_preds(model, val_loader)

    # Per-class Average Precision
    ap_scores = []
    print(f"\n{'Class':<15s}  {'AP':>6s}")
    print("-" * 25)
    for i, label in enumerate(LABELS):
        ap = average_precision_score(labels[:, i], probs[:, i])
        ap_scores.append(ap)
        print(f"{label:<15s}  {ap:6.4f}")

    map_score = np.mean(ap_scores)
    print("-" * 25)
    print(f"{'mAP':<15s}  {map_score:6.4f}")

    # Best thresholds
    best_thresholds = find_best_thresholds(probs, labels)
    print("\nBest thresholds per class (for predict step):")
    for label, t in zip(LABELS, best_thresholds):
        print(f"  {label:<15s}: {t:.2f}")

    # Save thresholds for use by 06_predict.py
    threshold_path = OUTPUT_DIR / "best_thresholds.npy"
    np.save(threshold_path, best_thresholds)
    print(f"\nThresholds saved to {threshold_path}")

    # Plot per-class AP bar chart
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(LABELS, ap_scores, color="steelblue")
    ax.axhline(map_score, color="red", linestyle="--", label=f"mAP={map_score:.3f}")
    ax.set_ylabel("Average Precision")
    ax.set_title("Per-class AP on validation set")
    ax.set_xticklabels(LABELS, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "eval_ap_per_class.png"
    plt.savefig(plot_path, dpi=100)
    plt.close()
    print(f"AP chart saved to {plot_path}")


if __name__ == "__main__":
    main()
