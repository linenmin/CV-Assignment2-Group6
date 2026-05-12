"""Re-split ``data/segman_ga2`` to a smaller validation ratio.

The ``data/segman_ga2`` PNG export currently uses the same 0.15 ratio as
``data/splits/`` (637 train / 112 val). For the V17 final fine-tune we
want to give the model more training data, so we re-split at a smaller
ratio and physically move the affected PNGs between the ``training/``
and ``validation/`` subdirectories.

The new split is a strict prefix of the original Python-``random.Random(42)``
shuffle. With ``--val-ratio 0.05`` the new validation set is exactly the
first 37 shuffled IDs, which is a subset of the original 112; no image
ever needs to move ``training -> validation``, so the operation is purely
a one-way relabel.

Old split files are backed up to ``training_{suffix}.txt`` and
``validation_{suffix}.txt`` before being overwritten.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/segman_ga2"))
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=Path("kul-computer-vision-ga-2-2026/train/train_set.csv"),
    )
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--backup-suffix",
        type=str,
        default="v15split",
        help="Suffix used when renaming the existing split files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned moves without modifying anything.",
    )
    return parser.parse_args()


def stem_from_id(sample_id: int) -> str:
    return f"train_{sample_id}"


def id_from_stem(stem: str) -> int:
    return int(stem.split("_")[1])


def main() -> None:
    args = parse_args()
    data_root: Path = args.data_root
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    train_df = pd.read_csv(args.train_csv, index_col="Id")
    all_ids = list(train_df.index.astype(int))
    rng = random.Random(args.seed)
    shuffled = all_ids[:]
    rng.shuffle(shuffled)
    val_size = int(round(len(shuffled) * args.val_ratio))
    new_val_ids = set(shuffled[:val_size])
    new_train_ids = set(shuffled[val_size:])

    current_train_path = data_root / "training.txt"
    current_val_path = data_root / "validation.txt"

    current_train_stems = [
        line.strip()
        for line in current_train_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    current_val_stems = [
        line.strip()
        for line in current_val_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    val_to_train: list[str] = []
    for stem in current_val_stems:
        if id_from_stem(stem) in new_train_ids:
            val_to_train.append(stem)

    train_to_val: list[str] = []
    for stem in current_train_stems:
        if id_from_stem(stem) in new_val_ids:
            train_to_val.append(stem)

    print(
        f"Current split: train={len(current_train_stems)} val={len(current_val_stems)}",
        flush=True,
    )
    print(
        f"New     split: train={len(new_train_ids)} val={len(new_val_ids)} "
        f"(val_ratio={args.val_ratio})",
        flush=True,
    )
    print(f"Moves val -> train : {len(val_to_train)}", flush=True)
    print(f"Moves train -> val : {len(train_to_val)}", flush=True)

    if args.dry_run:
        print("Dry run; not modifying anything.", flush=True)
        return

    backup_train = current_train_path.with_name(f"training_{args.backup_suffix}.txt")
    backup_val = current_val_path.with_name(f"validation_{args.backup_suffix}.txt")
    if backup_train.exists() or backup_val.exists():
        raise FileExistsError(
            f"Backup files already exist: {backup_train} / {backup_val}; "
            "pick a different --backup-suffix or remove them."
        )
    shutil.copy2(current_train_path, backup_train)
    shutil.copy2(current_val_path, backup_val)
    print(f"Backed up to {backup_train.name} and {backup_val.name}", flush=True)

    for stem in val_to_train:
        for subdir in ("images", "annotations"):
            src = data_root / subdir / "validation" / f"{stem}.png"
            dst = data_root / subdir / "training" / f"{stem}.png"
            if not src.exists():
                raise FileNotFoundError(f"Source PNG missing: {src}")
            if dst.exists():
                raise FileExistsError(f"Destination already occupied: {dst}")
            shutil.move(str(src), str(dst))

    for stem in train_to_val:
        for subdir in ("images", "annotations"):
            src = data_root / subdir / "training" / f"{stem}.png"
            dst = data_root / subdir / "validation" / f"{stem}.png"
            if not src.exists():
                raise FileNotFoundError(f"Source PNG missing: {src}")
            if dst.exists():
                raise FileExistsError(f"Destination already occupied: {dst}")
            shutil.move(str(src), str(dst))

    new_train_stems = sorted([stem_from_id(i) for i in new_train_ids], key=id_from_stem)
    new_val_stems = sorted([stem_from_id(i) for i in new_val_ids], key=id_from_stem)
    current_train_path.write_text("\n".join(new_train_stems) + "\n", encoding="utf-8")
    current_val_path.write_text("\n".join(new_val_stems) + "\n", encoding="utf-8")

    final_train_files = len(list((data_root / "images" / "training").glob("*.png")))
    final_val_files = len(list((data_root / "images" / "validation").glob("*.png")))
    print(
        f"Wrote new split. Files on disk: "
        f"training={final_train_files}, validation={final_val_files}",
        flush=True,
    )


if __name__ == "__main__":
    main()
