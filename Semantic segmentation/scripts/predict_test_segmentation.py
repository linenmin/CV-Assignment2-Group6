import argparse
from pathlib import Path

from ga2_seg.inference import save_test_predictions
from ga2_seg.paths import get_project_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict semantic segmentation masks for the test set.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint used for test inference.")
    parser.add_argument("--config", type=str, default=None, help="Experiment config path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=get_project_root() / "outputs" / "predictions" / "test_segmentation",
        help="Directory where per-image test masks will be stored.",
    )
    parser.add_argument("--device", type=str, default=None, help="Inference device, e.g. cuda:0 or cpu.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for quick smoke runs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = save_test_predictions(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        config_path=args.config,
        device=args.device,
        limit=args.limit,
    )
    print(f"prediction_dir={args.output_dir}")
    print(f"predictions_written={len(paths)}")


if __name__ == "__main__":
    main()
