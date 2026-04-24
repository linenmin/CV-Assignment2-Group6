from __future__ import annotations

from pathlib import Path

import mmcv.parallel._functions as mmcv_functions


PATCH_MARKER = "# GA2 torch>=2.1 compatibility patch"


def main() -> None:
    target = Path(mmcv_functions.__file__)
    text = target.read_text(encoding="utf-8")
    if PATCH_MARKER in text:
        print(f"already_patched={target}")
        return

    old = """        if input_device == -1 and target_gpus != [-1]:
            # Perform CPU to GPU copies in a background stream
            streams = [_get_stream(device) for device in target_gpus]
"""
    new = """        if input_device == -1 and target_gpus != [-1]:
            # Perform CPU to GPU copies in a background stream.
            # GA2 torch>=2.1 compatibility patch: PyTorch now expects
            # torch.device objects in _get_stream, while MMCV 1.x passes ints.
            import torch
            stream_gpus = [
                torch.device("cuda", device) if isinstance(device, int) else device
                for device in target_gpus
            ]
            streams = [_get_stream(device) for device in stream_gpus]
"""
    if old not in text:
        raise RuntimeError(f"Expected MMCV scatter block was not found in {target}")

    target.write_text(text.replace(old, new), encoding="utf-8")
    print(f"patched={target}")


if __name__ == "__main__":
    main()
