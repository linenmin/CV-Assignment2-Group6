import os
import sys
from collections import OrderedDict
from importlib import import_module


def configure_windows_runtime() -> None:
    """Apply conservative process-level guards for Windows scientific stacks."""
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("MPLBACKEND", "Agg")


def patch_mmengine_collect_env() -> None:
    """Patch MMEngine's Windows compiler probe when locale decoding breaks."""
    runner_module = import_module("mmengine.runner.runner")

    if getattr(runner_module.collect_env, "__name__", "") == "_safe_collect_env":
        return

    original_collect_env = runner_module.collect_env

    def _safe_collect_env():
        try:
            return original_collect_env()
        except UnicodeDecodeError:
            env_info = OrderedDict()
            env_info["sys.platform"] = sys.platform
            env_info["Python"] = sys.version.replace("\n", "")
            env_info["collect_env_warning"] = (
                "Skipped full MMEngine environment collection due to a Windows "
                "locale decode issue while probing the compiler toolchain."
            )
            return env_info

    collect_env_module = import_module("mmengine.utils.dl_utils.collect_env")
    collect_env_module.collect_env = _safe_collect_env
    runner_module.collect_env = _safe_collect_env
