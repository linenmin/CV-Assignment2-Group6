"""Utilities for the GA2 semantic segmentation project."""

from .mmseg_runtime import load_config, register_project_modules, resolve_config_path
from .paths import get_dataset_root, get_project_root

__all__ = [
    "get_dataset_root",
    "get_project_root",
    "load_config",
    "register_project_modules",
    "resolve_config_path",
]
