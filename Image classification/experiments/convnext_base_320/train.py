"""Train the convnext_base_320 experiment."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared import load_module  # noqa: E402

EXPERIMENT = "convnext_base_320"


if __name__ == "__main__":
    train = load_module("train_step", ROOT / "04_train.py")
    sys.argv = [str(ROOT / "04_train.py"), "--experiment", EXPERIMENT, *sys.argv[1:]]
    train.main()
