"""Light loader for the team's ConvNeXt-Small classifier checkpoint.

We do not rebuild the full ``MultiLabelClassifier`` here. The notebook
only proves that the bundled ``.pth`` file loads as a state_dict and
reports its parameter count, which is enough to convince the reader the
artifact is real without dragging the whole torchvision backbone import
into Chapter 2.
"""

from __future__ import annotations

from pathlib import Path

import torch


def load_convnext_small_weights(path: str | Path) -> dict:
    """Load the state_dict from a ``.pth`` file and return a small summary.

    Returns
    -------
    dict
        Keys: ``param_count`` (int), ``layer_count`` (int),
        ``head_keys`` (list of strings, the keys in the classifier head),
        ``path`` (resolved path), ``state_dict_keys_total`` (int).
    """

    state_dict = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    param_count = int(sum(t.numel() for t in state_dict.values() if torch.is_tensor(t)))
    head_keys = [k for k in state_dict.keys() if "head" in k or "classifier" in k]
    return {
        "path": str(Path(path).resolve()),
        "param_count": param_count,
        "layer_count": len({k.rsplit(".", 1)[0] for k in state_dict.keys()}),
        "head_keys": head_keys[:6],
        "state_dict_keys_total": len(state_dict.keys()),
    }
