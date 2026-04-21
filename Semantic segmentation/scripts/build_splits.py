import argparse

from ga2_seg.split import (
    build_train_val_split,
    get_train_split_path,
    get_val_split_path,
    write_split_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train/val splits for semantic segmentation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    train_ids, val_ids = build_train_val_split(seed=args.seed, val_ratio=args.val_ratio)
    train_path = write_split_file(get_train_split_path(), train_ids)
    val_path = write_split_file(get_val_split_path(), val_ids)

    print(f"train_split={train_path}")
    print(f"val_split={val_path}")
    print(f"train_count={len(train_ids)}")
    print(f"val_count={len(val_ids)}")


if __name__ == "__main__":
    main()
