"""Utilities for the GA2 semantic segmentation project."""

from importlib import import_module

from .paths import get_dataset_root, get_project_root


def __getattr__(name: str):
    if name in {"build_submission_dataframe", "write_submission_csv"}:
        submission = import_module(".submission", __name__)

        return getattr(submission, name)
    if name in {"build_training_summary", "load_scalar_records"}:
        training_analysis = import_module(".training_analysis", __name__)

        return getattr(training_analysis, name)
    if name in {"load_config", "register_project_modules", "resolve_config_path"}:
        mmseg_runtime = import_module(".mmseg_runtime", __name__)

        return getattr(mmseg_runtime, name)
    if name == "DelayedEarlyStoppingHook":
        early_stopping = import_module(".early_stopping", __name__)

        return getattr(early_stopping, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "build_submission_dataframe",
    "build_training_summary",
    "DelayedEarlyStoppingHook",
    "get_dataset_root",
    "get_project_root",
    "load_scalar_records",
    "load_config",
    "register_project_modules",
    "resolve_config_path",
    "write_submission_csv",
]
