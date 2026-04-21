from pathlib import Path

import random
import pandas as pd

from .paths import get_project_root
from .stats import load_train_labels_df


def get_data_dir() -> Path:
    return get_project_root() / "data"


def get_split_dir() -> Path:
    return get_data_dir() / "splits"


def get_train_split_path() -> Path:
    return get_split_dir() / "train.txt"


def get_val_split_path() -> Path:
    return get_split_dir() / "val.txt"


def build_train_val_split(seed: int = 42, val_ratio: float = 0.15) -> tuple[list[int], list[int]]:
    train_df = load_train_labels_df()
    all_ids = list(train_df.index.astype(int))
    rng = random.Random(seed)
    shuffled_ids = all_ids[:]
    rng.shuffle(shuffled_ids)
    val_size = int(round(len(shuffled_ids) * val_ratio))
    val_ids = sorted(shuffled_ids[:val_size])
    train_ids = sorted(shuffled_ids[val_size:])
    return train_ids, val_ids


def write_split_file(path: Path, sample_ids: list[int]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(sample_id) for sample_id in sample_ids) + "\n", encoding="utf-8")
    return path
