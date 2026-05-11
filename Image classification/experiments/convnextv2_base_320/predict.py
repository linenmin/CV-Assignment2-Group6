"""Predict the test set with the convnextv2_base_320 experiment."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared import load_module  # noqa: E402

EXPERIMENT = "convnextv2_base_320"


if __name__ == "__main__":
    predict = load_module("predict_step", ROOT / "06_predict.py")
    sys.argv = [str(ROOT / "06_predict.py"), "--experiment", EXPERIMENT, *sys.argv[1:]]
    predict.main()
