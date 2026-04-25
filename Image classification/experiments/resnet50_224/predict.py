"""Predict the test set with the resnet50_224 experiment."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared import get_experiment_config, load_module  # noqa: E402

EXPERIMENT = "resnet50_224"


if __name__ == "__main__":
    predict = load_module("predict_step", ROOT / "06_predict.py")
    sys.argv = [str(ROOT / "06_predict.py"), "--experiment", EXPERIMENT, *sys.argv[1:]]
    predict.main()
