"""RLE encoding and decoding for the Kaggle submission format.

The encoder is a direct port of ``_rle_encode`` from the starter notebook
``ga2_group_6.ipynb``. It accepts any binary array, flattens it in
row-major order, and emits a space-separated ``start length start length
...`` string. The decoder is the matching inverse; it is included so the
notebook can demonstrate the round-trip without depending on any model.
"""

from __future__ import annotations

import numpy as np


def rle_encode(mask: np.ndarray) -> str:
    """Run-length-encode a binary mask into the Kaggle submission string.

    The convention matches the starter notebook: pixels are read in
    row-major order, runs are 1-indexed, and the output alternates
    ``start length`` pairs.

    Parameters
    ----------
    mask : np.ndarray
        Binary array. Non-zero entries are treated as foreground.

    Returns
    -------
    str
        The space-separated RLE string. Empty string if the mask has no
        foreground pixels.
    """

    pixels = np.asarray(mask, dtype=np.uint8).flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(int(x)) for x in runs)


def rle_decode(rle_string: str, shape: tuple[int, ...]) -> np.ndarray:
    """Decode an RLE string back into a binary array of the given shape.

    Parameters
    ----------
    rle_string : str
        The RLE string produced by :func:`rle_encode`.
    shape : tuple[int, ...]
        Target shape; the total number of elements must match the
        original mask.

    Returns
    -------
    np.ndarray
        Binary array of the requested shape, dtype ``uint8``.
    """

    total = int(np.prod(shape))
    flat = np.zeros(total, dtype=np.uint8)
    if not rle_string:
        return flat.reshape(shape)
    tokens = [int(token) for token in rle_string.split()]
    starts = np.asarray(tokens[0::2], dtype=np.int64) - 1
    lengths = np.asarray(tokens[1::2], dtype=np.int64)
    for start, length in zip(starts, lengths):
        flat[start : start + length] = 1
    return flat.reshape(shape)
