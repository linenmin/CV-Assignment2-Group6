"""
Step 6: Run inference on the test set and generate submission.csv.

Run from project root:
    python "Image classification/06_predict.py"

Output:
    output/submission.csv   (Kaggle RLE format for classification rows)

NOTE: This script only fills classification columns. Segmentation predictions
      must be added separately (Section 2 of the notebook) before final submission.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from shared import LABELS, DATA_DIR, OUTPUT_DIR, NUM_WORKERS, load_module  # noqa: E402

_here = Path(__file__).parent
_ds   = load_module("dataset", _here / "02_dataset.py")
_mdl  = load_module("model",   _here / "03_model.py")

VOCDataset            = _ds.VOCDataset
load_test_df          = _ds.load_test_df
get_val_transform     = _ds.get_val_transform
MultiLabelClassifier  = _mdl.MultiLabelClassifier

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BACKBONE  = "efficientnet_b3"   # must match what was used in 04_train.py
BATCH_SIZE = 16
# Use final_model.pth (all-data retrain) if available, else best_model.pth
CKPT_PATH         = OUTPUT_DIR / "checkpoints" / "final_model.pth"
CKPT_FALLBACK     = OUTPUT_DIR / "checkpoints" / "best_model.pth"
THRESH_PATH       = OUTPUT_DIR / "best_thresholds.npy"
DEFAULT_THRESHOLD = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# TTA: average predictions over original + horizontal flip
# ---------------------------------------------------------------------------
def _predict_tta(model: torch.nn.Module, imgs: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        p1 = torch.sigmoid(model(imgs))
        p2 = torch.sigmoid(model(imgs.flip(-1)))   # horizontal flip
    return (p1 + p2) / 2


# ---------------------------------------------------------------------------
# RLE helper (mirrors generate_submission from the notebook template)
# ---------------------------------------------------------------------------
def _rle_encode(arr: np.ndarray) -> str:
    pixels = arr.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs   = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(ckpt_path=None):
    path = Path(ckpt_path) if ckpt_path else (
        CKPT_PATH if CKPT_PATH.exists() else CKPT_FALLBACK
    )
    if not path.exists():
        print(f"Checkpoint not found: {path}")
        print("Run 04_train.py first.")
        return
    print(f"Loading checkpoint: {path.name}")

    # Load per-class thresholds from evaluate step (fall back to 0.5)
    if THRESH_PATH.exists():
        thresholds = np.load(THRESH_PATH)
        print(f"Loaded per-class thresholds from {THRESH_PATH}")
    else:
        thresholds = np.full(len(LABELS), DEFAULT_THRESHOLD)
        print(f"Using default threshold {DEFAULT_THRESHOLD} for all classes.")

    # Load model
    model = MultiLabelClassifier(backbone=BACKBONE, num_classes=len(LABELS),
                                  pretrained=False).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()

    # Build test loader
    test_df = load_test_df(DATA_DIR)
    test_ds = VOCDataset(test_df, DATA_DIR, split="test",
                         transform=get_val_transform())
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    # Inference with TTA (original + horizontal flip) — +0.5–1% at zero cost
    all_probs = []
    for imgs, _ in tqdm(test_loader, desc="Predicting (TTA)"):
        probs = _predict_tta(model, imgs.to(device)).cpu().numpy()
        all_probs.append(probs)

    all_probs = np.vstack(all_probs)                       # (750, 20)
    preds     = (all_probs > thresholds[None, :]).astype(int)

    # Fill test dataframe classification columns
    test_df[LABELS] = preds

    # -------------------------------------------------------------------
    # Build submission rows: classification (RLE) + segmentation (empty)
    # Interleaved: {id}_classification, {id}_segmentation, ...
    # -------------------------------------------------------------------
    rows = {"Id": [], "Predicted": []}
    for idx in test_df.index:
        label_vec = test_df.loc[idx, LABELS].values.astype(int)
        rows["Id"].append(f"{idx}_classification")
        rows["Predicted"].append(_rle_encode(label_vec))
        rows["Id"].append(f"{idx}_segmentation")
        rows["Predicted"].append("")

    submission_df = pd.DataFrame(rows).set_index("Id")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "submission_classification.csv"
    submission_df.to_csv(out_path)
    print(f"\nSubmission saved to {out_path}")
    print(f"Rows: {len(submission_df)}  (classification + empty segmentation)")
    print("\nFirst 4 rows:")
    print(submission_df.head(4))

    # Also return the filled test_df for downstream use (segmentation step)
    return test_df


if __name__ == "__main__":
    main()
