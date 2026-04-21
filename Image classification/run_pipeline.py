"""
Full pipeline: explore → train → evaluate → predict.

Run from project root:
    python "Image classification/run_pipeline.py"

Common usage:
    # Full run from scratch
    python "Image classification/run_pipeline.py"

    # PNGs already exported, retrain from scratch
    python "Image classification/run_pipeline.py" --skip-explore

    # Skip both explore and train, use existing best checkpoint
    python "Image classification/run_pipeline.py" --skip-explore --skip-train

    # Skip train, use last checkpoint instead of best
    python "Image classification/run_pipeline.py" --skip-explore --skip-train --ckpt last
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from shared import OUTPUT_DIR, load_module

_here = Path(__file__).parent
CKPT_DIR = OUTPUT_DIR / "checkpoints"


def _sep(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-explore", action="store_true",
                        help="Skip step 1 (PNG export already done)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip step 2 (use existing checkpoint)")
    parser.add_argument("--ckpt", choices=["best", "last"], default="best",
                        help="Which checkpoint to use for eval/predict (default: best)")
    args = parser.parse_args()

    ckpt_path = CKPT_DIR / f"{args.ckpt}_model.pth"

    if not args.skip_explore:
        _sep("Step 1 / 4 — Data exploration + PNG export")
        load_module("explore", _here / "01_explore_data.py").main()

    if not args.skip_train:
        _sep("Step 2 / 4 — Training (two-stage ResNet-50 fine-tune)")
        load_module("train", _here / "04_train.py").main()
    else:
        print(f"\n[skip-train] Using checkpoint: {ckpt_path}")

    _sep("Step 3 / 4 — Evaluation (mAP + best thresholds)")
    load_module("evaluate", _here / "05_evaluate.py").main(ckpt_path=ckpt_path)

    _sep("Step 4 / 4 — Predict test set + generate submission")
    load_module("predict", _here / "06_predict.py").main(ckpt_path=ckpt_path)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
