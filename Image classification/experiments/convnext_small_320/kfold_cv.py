"""Run 5-fold CV for the convnext_small_320 experiment."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared import load_module  # noqa: E402

EXPERIMENT = "convnext_small_320"

if __name__ == "__main__":
    kfold = load_module("kfold_cv_step", ROOT / "09_kfold_cv.py")
    sys.argv = [str(ROOT / "09_kfold_cv.py"), "--experiment", EXPERIMENT, *sys.argv[1:]]
    kfold.main()
