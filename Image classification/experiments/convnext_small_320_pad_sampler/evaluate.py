"""Evaluate the convnext_small_320_pad_sampler experiment."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared import load_module  # noqa: E402

EXPERIMENT = "convnext_small_320_pad_sampler"


if __name__ == "__main__":
    evaluate = load_module("evaluate_step", ROOT / "05_evaluate.py")
    sys.argv = [str(ROOT / "05_evaluate.py"), "--experiment", EXPERIMENT, *sys.argv[1:]]
    evaluate.main()
