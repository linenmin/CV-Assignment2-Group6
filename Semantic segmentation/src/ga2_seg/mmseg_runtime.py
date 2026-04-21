from pathlib import Path

from mmengine.config import Config
from mmengine.runner import Runner
from mmseg.registry import DATASETS
from mmseg.utils import register_all_modules

from .paths import get_project_root
from .runtime_compat import patch_mmengine_collect_env


DEFAULT_CONFIG_PATH = (
    get_project_root()
    / "configs"
    / "experiments"
    / "segnext_s_512x512_adamw_poly_v1.py"
)


def register_project_modules() -> None:
    """Register MMSeg and project-specific modules before building configs."""
    patch_mmengine_collect_env()
    register_all_modules()

    # Import custom dataset and transforms so custom_imports are not the only
    # registration path. This keeps tests and local scripts consistent.
    import ga2_seg.early_stopping  # noqa: F401
    import ga2_seg.mmseg_dataset  # noqa: F401
    import ga2_seg.mmseg_transforms  # noqa: F401
    import ga2_seg.region_rebalance_head  # noqa: F401


def resolve_config_path(config_path: str | Path | None = None) -> Path:
    if config_path is None:
        return DEFAULT_CONFIG_PATH

    path = Path(config_path)
    if not path.is_absolute():
        path = get_project_root() / path
    return path.resolve()


def load_config(config_path: str | Path | None = None) -> Config:
    register_project_modules()
    return Config.fromfile(str(resolve_config_path(config_path)))


def build_dataset(cfg: Config, split: str = "train"):
    split_to_key = {
        "train": "train_dataloader",
        "val": "val_dataloader",
        "test": "test_dataloader",
    }
    if split not in split_to_key:
        raise ValueError(f"Unsupported split: {split}")

    register_project_modules()
    dataset_cfg = getattr(cfg, split_to_key[split]).dataset
    return DATASETS.build(dataset_cfg)


def build_runner(config_path: str | Path | None = None) -> Runner:
    cfg = load_config(config_path)
    return Runner.from_cfg(cfg)
