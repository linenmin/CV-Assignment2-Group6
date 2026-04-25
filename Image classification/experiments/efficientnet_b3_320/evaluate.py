"""Evaluate the efficientnet_b3_320 experiment."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared import get_experiment_config, load_module  # noqa: E402

EXPERIMENT = "efficientnet_b3_320"


if __name__ == "__main__":
    evaluate = load_module("evaluate_step", ROOT / "05_evaluate.py")
    sys.argv = [str(ROOT / "05_evaluate.py"), "--experiment", EXPERIMENT, *sys.argv[1:]]
    evaluate.main()
