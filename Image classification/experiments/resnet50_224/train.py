"""Train the resnet50_224 experiment."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared import get_experiment_config, load_module  # noqa: E402

EXPERIMENT = "resnet50_224"


if __name__ == "__main__":
    train = load_module("train_step", ROOT / "04_train.py")
    sys.argv = [str(ROOT / "04_train.py"), "--experiment", EXPERIMENT, *sys.argv[1:]]
    train.main()
