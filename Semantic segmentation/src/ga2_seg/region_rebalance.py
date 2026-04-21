from pathlib import Path

from .labels import CLASS_NAMES
from .paths import get_project_root
from .stats import load_train_labels_df


REGION_REBALANCE_LOSS_WEIGHT = 0.5


def _resolve_split_path(split_path: Path | None = None) -> Path:
    if split_path is not None:
        return Path(split_path)
    return get_project_root() / "data" / "splits" / "train.txt"


def load_split_ids(split_path: Path | None = None) -> list[int]:
    resolved_path = _resolve_split_path(split_path)
    return [int(line.strip()) for line in resolved_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_region_class_frequencies(split_path: Path | None = None) -> list[int]:
    sample_ids = load_split_ids(split_path)
    train_df = load_train_labels_df()
    split_df = train_df.loc[sample_ids, list(CLASS_NAMES)]
    return [int(split_df[class_name].sum()) for class_name in CLASS_NAMES]
