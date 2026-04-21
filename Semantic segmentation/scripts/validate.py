import argparse
from pathlib import Path

from mmengine.runner import Runner

from ga2_seg.mmseg_runtime import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate GA2 semantic segmentation checkpoint.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Experiment config path. Defaults to the approved SegNeXt-S config.",
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path to evaluate.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override validation workers.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg.load_from = str(Path(args.checkpoint).resolve())

    if "checkpoint" in cfg.default_hooks:
        cfg.default_hooks.checkpoint.save_best = None
        cfg.default_hooks.checkpoint.interval = -1

    if args.num_workers is not None:
        cfg.val_dataloader.num_workers = args.num_workers
        cfg.val_dataloader.persistent_workers = args.num_workers > 0

    runner = Runner.from_cfg(cfg)
    runner.val()


if __name__ == "__main__":
    main()
