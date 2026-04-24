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
# Asymmetric Loss for Multi-Label Classification (ICCV 2021)
# Ridnik et al.  https://arxiv.org/abs/2009.14119
#
# Two-pronged attack on the false-negative noise problem in PASCAL VOC:
#
#   1. Probability shifting (clip):
#      For negatives, replace p with max(p - clip, 0) before log.
#      Any prediction below 'clip' contributes zero loss — the model is
#      not punished for being slightly positive on unlabelled objects.
#
#   2. Asymmetric focal decay:
#      γ_pos=0 (no down-weighting on positives, every TP counts).
#      γ_neg=4 (aggressively down-weight easy negatives so rare positives
#               dominate the gradient).
#
# Result: diningtable, pottedplant, and other noisy/rare classes
# receive proportionally stronger gradient signal.
# ---------------------------------------------------------------------------
class AsymmetricLoss(nn.Module):
    """
    Parameters
    ----------
    gamma_neg : float  Focal exponent for negatives (paper default: 4).
    gamma_pos : float  Focal exponent for positives (paper default: 0).
    clip      : float  Probability margin for negatives (paper default: 0.05).
    """

    def __init__(self, gamma_neg: float = 4, gamma_pos: float = 0,
                 clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs     = torch.sigmoid(logits)
        probs_neg = 1.0 - probs

        # Shift negatives: zero loss when predicted prob < clip
        if self.clip > 0:
            probs_neg = (probs_neg + self.clip).clamp(max=1.0)

        log_p  = torch.log(probs.clamp(min=self.eps))
        log_np = torch.log(probs_neg.clamp(min=self.eps))
        loss   = targets * log_p + (1 - targets) * log_np

        # Asymmetric focal weights (detached — weights are not optimised)
        with torch.no_grad():
            w = (targets       * (1 - probs).pow(self.gamma_pos) +
                 (1 - targets) * probs.pow(self.gamma_neg))
        loss = loss * w

        return -loss.mean()
