"""Run explore, train, evaluate, and predict for one experiment."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from shared import EXPERIMENTS, describe_experiment, get_experiment_config, load_module  # noqa: E402

_here = Path(__file__).parent


def sep(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        choices=sorted(EXPERIMENTS),
        default=None,
        help="Experiment config to run.",
    )
    parser.add_argument("--skip-explore", action="store_true", help="Skip data exploration.")
    parser.add_argument("--skip-train", action="store_true", help="Use an existing checkpoint.")
    parser.add_argument(
        "--ckpt",
        choices=["best", "final", "last"],
        default="best",
        help="Checkpoint for evaluate/predict when training is skipped.",
    )
    return parser.parse_args()


def main(config=None):
    args = parse_args() if config is None else None
    config = config or get_experiment_config(args.experiment)

    skip_explore = args.skip_explore if args else False
    skip_train = args.skip_train if args else False
    ckpt_name = args.ckpt if args else "best"

    print(f"Experiment: {describe_experiment(config)}")

    if not skip_explore:
        sep("Step 1 / 4 - data exploration")
        load_module("explore", _here / "01_explore_data.py").main()

    if not skip_train:
        sep("Step 2 / 4 - training")
        load_module("train", _here / "04_train.py").main(config=config)
        ckpt_name = "best"
    else:
        print(f"\n[skip-train] Using {ckpt_name} checkpoint from {config.checkpoints_dir}")

    sep("Step 3 / 4 - evaluation")
    load_module("evaluate", _here / "05_evaluate.py").main(
        config=config,
        ckpt_name=ckpt_name,
    )

    sep("Step 4 / 4 - predict test set + submission CSV")
    predict_ckpt = "auto" if not skip_train else ckpt_name
    load_module("predict", _here / "06_predict.py").main(
        config=config,
        ckpt_name=predict_ckpt,
    )

    print("\nPipeline complete.")
    print(f"Experiment output: {config.output_dir}")


if __name__ == "__main__":
    main()
