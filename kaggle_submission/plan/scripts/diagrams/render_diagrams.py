"""Render the HTML diagram files to PNG via headless Chrome, then crop to
the actual content box and save into dataset/figures/ for the notebook
to embed as static images.

The notebook itself does not render mermaid in Jupyter / Kaggle (the
classic notebook viewer has no mermaid support), so we ship the
diagrams as committed PNG files instead.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[3]
FIGURES_DIR = PROJECT_ROOT / "kaggle_submission" / "dataset" / "figures"

CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")

DIAGRAMS = [
    ("ch3_eomt", HERE / "ch3_eomt.html", FIGURES_DIR / "ch3_eomt_dataflow.png", (1480, 720)),
    ("ch4_attack_modes", HERE / "ch4_attack_modes.html", FIGURES_DIR / "ch4_attack_modes.png", (1480, 720)),
]


def render_one(name: str, html_path: Path, out_path: Path, window: tuple[int, int]) -> None:
    if not CHROME.is_file():
        raise SystemExit(f"Chrome not found at {CHROME}")
    if not html_path.is_file():
        raise SystemExit(f"Source HTML missing: {html_path}")

    raw = html_path.with_suffix(".raw.png")
    cmd = [
        str(CHROME),
        "--headless",
        "--disable-gpu",
        "--hide-scrollbars",
        f"--window-size={window[0]},{window[1]}",
        "--default-background-color=00000000",
        f"--screenshot={raw}",
        html_path.as_uri(),
    ]
    print(f"[{name}] rendering {html_path.name} -> {raw.name}", flush=True)
    subprocess.run(cmd, check=True, capture_output=True)

    # Crop tight to the actual content box (drop bottom whitespace).
    img = Image.open(raw)
    arr = np.array(img.convert("L"))
    non_white = (arr < 250)
    if non_white.any():
        rows = np.where(non_white.any(axis=1))[0]
        cols = np.where(non_white.any(axis=0))[0]
        # Add a small margin so text edges are not clipped.
        margin = 8
        top = max(0, int(rows.min()) - margin)
        bot = min(img.height, int(rows.max()) + margin)
        left = max(0, int(cols.min()) - margin)
        right = min(img.width, int(cols.max()) + margin)
        img = img.crop((left, top, right, bot))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, optimize=True)
    print(f"[{name}] wrote {out_path}  size={img.size}", flush=True)

    raw.unlink(missing_ok=True)


def main() -> None:
    for name, src, dst, window in DIAGRAMS:
        render_one(name, src, dst, window)


if __name__ == "__main__":
    main()
