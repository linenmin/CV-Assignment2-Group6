import argparse
from pathlib import Path

from ga2_seg.segman_adapter import export_segman_dataset, write_segman_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare GA2 data and configs for official SegMAN training.")
    parser.add_argument(
        "--segman-root",
        type=Path,
        required=True,
        help="Path to the cloned official SegMAN repository.",
    )
    parser.add_argument(
        "--variant",
        choices=("b", "l"),
        default="b",
        help="SegMAN model size to configure.",
    )
    parser.add_argument(
        "--encoder-checkpoint",
        type=Path,
        required=True,
        help="Path to the ImageNet-pretrained SegMAN encoder checkpoint.",
    )
    parser.add_argument("--max-iters", type=int, default=30000)
    parser.add_argument("--val-interval", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite exported PNG files.")
    parser.add_argument("--test-limit", type=int, default=None, help="Optional quick export limit for test images.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_result = export_segman_dataset(
        overwrite=args.overwrite,
        test_limit=args.test_limit,
    )
    config_path = write_segman_config(
        segman_root=args.segman_root,
        variant=args.variant,
        encoder_checkpoint=args.encoder_checkpoint,
        max_iters=args.max_iters,
        val_interval=args.val_interval,
        batch_size=args.batch_size,
        workers=args.workers,
    )

    print(f"segman_export_root={export_result.export_root}")
    print(f"train_count={export_result.train_count}")
    print(f"val_count={export_result.val_count}")
    print(f"test_count={export_result.test_count}")
    print(f"segman_config={config_path}")
    segmentation_root = args.segman_root / "segmentation"
    display_config_path = (
        config_path.relative_to(segmentation_root)
        if config_path.is_relative_to(segmentation_root)
        else config_path
    )
    print(
        "train_command="
        f"cd {segmentation_root} && "
        f"python tools/train.py {display_config_path} --work-dir outputs/ga2_segman_{args.variant}"
    )


if __name__ == "__main__":
    main()
