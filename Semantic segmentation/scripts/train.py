import argparse
from pathlib import Path

from mmengine.runner import Runner

from ga2_seg.checkpoint_cleanup import prune_checkpoints
from ga2_seg.mmseg_runtime import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GA2 semantic segmentation model.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Experiment config path. Defaults to the approved SegNeXt-S config.",
    )
    parser.add_argument("--work-dir", type=str, default=None, help="Override work directory.")
    parser.add_argument("--max-iters", type=int, default=None, help="Override total iterations.")
    parser.add_argument("--val-interval", type=int, default=None, help="Override validation interval.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override train batch size.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override data loader workers for train and validation.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable the default ADE20K checkpoint for smoke tests.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume training from work_dir.")
    parser.add_argument(
        "--checkpoint-retention",
        choices=["best_last", "all"],
        default="best_last",
        help="Checkpoint retention policy after training finishes.",
    )
    return parser.parse_args()


def _set_loader_num_workers(loader_cfg, num_workers: int) -> None:
    loader_cfg.num_workers = num_workers
    loader_cfg.persistent_workers = num_workers > 0


def _set_max_iters(cfg, max_iters: int) -> None:
    cfg.train_cfg.max_iters = max_iters
    if "default_hooks" in cfg and "checkpoint" in cfg.default_hooks:
        cfg.default_hooks.checkpoint.interval = min(
            cfg.default_hooks.checkpoint.interval,
            max_iters,
        )

    if "param_scheduler" not in cfg:
        return

    warmup_end = min(cfg.param_scheduler[0].end, max_iters)
    cfg.param_scheduler[0].end = warmup_end

    if max_iters <= warmup_end:
        cfg.param_scheduler = [cfg.param_scheduler[0]]
        return

    cfg.param_scheduler[1].begin = warmup_end
    cfg.param_scheduler[1].end = max_iters


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.work_dir:
        cfg.work_dir = str(Path(args.work_dir).resolve())
    if args.max_iters is not None:
        _set_max_iters(cfg, args.max_iters)
    if args.val_interval is not None:
        cfg.train_cfg.val_interval = args.val_interval
        cfg.default_hooks.checkpoint.interval = args.val_interval
    if args.batch_size is not None:
        cfg.train_dataloader.batch_size = args.batch_size
    if args.num_workers is not None:
        _set_loader_num_workers(cfg.train_dataloader, args.num_workers)
        _set_loader_num_workers(cfg.val_dataloader, args.num_workers)
    if args.no_pretrained:
        cfg.load_from = None

    cfg.resume = args.resume

    runner = Runner.from_cfg(cfg)
    runner.train()

    if args.checkpoint_retention == "best_last":
        result = prune_checkpoints(cfg.work_dir)
        print(f"checkpoint_retention={args.checkpoint_retention}")
        print(f"checkpoint_deleted={len(result['deleted'])}")
        print(f"checkpoint_freed_bytes={result['freed_bytes']}")


if __name__ == "__main__":
    main()
