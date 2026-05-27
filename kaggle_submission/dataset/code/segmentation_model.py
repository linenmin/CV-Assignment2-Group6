"""Light loader for the V17 EoMT-DINOv3 checkpoint.

The notebook only proves the bundled ``model.safetensors`` file is a
valid Hugging Face EoMT-DINOv3 state_dict at the expected scale; it
does not rebuild the full ``EomtDinov3ForUniversalSegmentation`` so
that Chapter 3 stays portable across Kaggle images that may or may not
have the matching ``transformers`` version cached.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_eomt_v17_summary(checkpoint_dir: str | Path) -> dict:
    """Inspect the bundled checkpoint without instantiating the model."""
    checkpoint_dir = Path(checkpoint_dir)
    config_path = checkpoint_dir / "config.json"
    metrics_path = checkpoint_dir / "metrics.json"
    safetensors_path = checkpoint_dir / "model.safetensors"

    summary: dict[str, object] = {"checkpoint_dir": str(checkpoint_dir.resolve())}

    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        summary["model_type"] = config.get("model_type", "?")
        summary["num_queries"] = config.get("num_queries", "?")
        summary["num_labels"] = len(config.get("id2label", {})) or "?"
        summary["hidden_size"] = config.get("hidden_size", "?")
        summary["patch_size"] = config.get("patch_size", "?")
        summary["image_size"] = config.get("image_size", "?")

    if safetensors_path.exists():
        # Read just the safetensors header to count tensors and bytes.
        with safetensors_path.open("rb") as handle:
            header_len = int.from_bytes(handle.read(8), "little")
            header_bytes = handle.read(header_len)
        header = json.loads(header_bytes)
        header.pop("__metadata__", None)
        param_count = 0
        for tensor_info in header.values():
            shape = tensor_info.get("shape", [])
            tensor_size = 1
            for dim in shape:
                tensor_size *= int(dim)
            param_count += tensor_size
        summary["state_dict_keys_total"] = len(header)
        summary["param_count"] = int(param_count)
        summary["safetensors_bytes"] = safetensors_path.stat().st_size

    if metrics_path.exists():
        summary["metrics_json"] = json.loads(metrics_path.read_text(encoding="utf-8"))

    return summary
