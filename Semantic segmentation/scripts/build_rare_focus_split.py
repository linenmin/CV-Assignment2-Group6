import argparse

from ga2_seg.rare_focus import RARE_FOCUS_CLASS_NAMES, RARE_FOCUS_DUPLICATE_FACTOR
from ga2_seg.split import (
    build_rare_focus_train_split,
    build_train_val_split,
    get_rare_focus_train_split_path,
    write_split_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the rare-focus training split for semantic segmentation V2."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--duplicate-factor", type=int, default=RARE_FOCUS_DUPLICATE_FACTOR)
    args = parser.parse_args()

    train_ids, _ = build_train_val_split(seed=args.seed, val_ratio=args.val_ratio)
    rare_focus_ids = build_rare_focus_train_split(
        base_train_ids=train_ids,
        duplicate_factor=args.duplicate_factor,
    )
    split_path = write_split_file(get_rare_focus_train_split_path(), rare_focus_ids)

    print(f"rare_focus_split={split_path}")
    print(f"base_train_count={len(train_ids)}")
    print(f"rare_focus_count={len(rare_focus_ids)}")
    print(f"focus_classes={','.join(RARE_FOCUS_CLASS_NAMES)}")
    print(f"duplicate_factor={args.duplicate_factor}")


if __name__ == "__main__":
    main()
