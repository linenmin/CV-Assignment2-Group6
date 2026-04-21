"""
Shared constants, utilities, and loss functions used across all scripts.
"""

import importlib.util
import os
from pathlib import Path
import torch
import torch.nn as nn

# Windows uses "spawn" multiprocessing: worker processes can't re-import
# modules that were loaded dynamically via importlib, so num_workers must be 0.
NUM_WORKERS = 0 if os.name == "nt" else 4

# ---------------------------------------------------------------------------
# Canonical class list (order matches train_set.csv columns)
# ---------------------------------------------------------------------------
LABELS = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

# ---------------------------------------------------------------------------
# Common paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / "kul-computer-vision-ga-2-2026"
OUTPUT_DIR   = PROJECT_ROOT / "output"


# ---------------------------------------------------------------------------
# Module loader — needed because script filenames start with digits,
# which prevents normal `import` statements from working.
# ---------------------------------------------------------------------------
def load_module(name: str, filepath: Path):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Noise-robust loss
#
# PASCAL VOC has systematic false-negative noise: objects are physically
# present in many images but not annotated (the "person in the background"
# problem visible in the sample grid).  Penalising the model hard for
# predicting those as positive hurts generalisation.
#
# Fix: smooth only the *negative* targets (0 → neg_smooth).
# Positive targets stay at 1.0 — VOC positive labels are reliable.
# ---------------------------------------------------------------------------
class NegativeSmoothBCELoss(nn.Module):
    """
    BCEWithLogitsLoss with one-sided label smoothing on negative targets.

    Parameters
    ----------
    neg_smooth : float
        Value to replace 0-targets with (e.g. 0.05 means "this class is
        absent with 95% confidence, not 100%").
    pos_weight : Tensor or None
        Per-class positive weights for class-imbalance compensation.
    """

    def __init__(self, neg_smooth: float = 0.05, pos_weight=None):
        super().__init__()
        self.neg_smooth = neg_smooth
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Only relax zeros; leave ones untouched
        soft = targets.clone()
        soft[targets == 0] = self.neg_smooth
        return self.bce(logits, soft)
