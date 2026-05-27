"""Zip the dataset directory into a single archive ready for Kaggle
dataset upload.

The script names the archive after the Kaggle slug used in the
notebook (`kul-cv-ga2-group6`) and writes a SHA-256 checksum next to
it so the user can confirm the file finished transferring.
"""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

ROOT = Path(
    r"D:\BaiduNetdiskWorkspace\Leuven\8th\Computer Vision\assignment\Group2\kaggle_submission"
)
DATASET = ROOT / "dataset"
OUT_DIR = ROOT / "outputs"
SLUG = "kul-cv-ga2-group6"


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    out_zip = OUT_DIR / f"{SLUG}.zip"
    if out_zip.exists():
        out_zip.unlink()

    files: list[Path] = sorted(p for p in DATASET.rglob("*") if p.is_file())
    total = 0
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_STORED) as zf:
        for p in files:
            arc = p.relative_to(DATASET).as_posix()
            zf.write(p, arc)
            total += p.stat().st_size
    print(f"Wrote {out_zip}")
    print(f"  files     : {len(files)}")
    print(f"  total in  : {total/1024/1024:.1f} MB")
    print(f"  zip out   : {out_zip.stat().st_size/1024/1024:.1f} MB")

    sha = hashlib.sha256()
    with out_zip.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)
    digest = sha.hexdigest()
    (out_zip.with_suffix(".zip.sha256")).write_text(
        f"{digest}  {out_zip.name}\n", encoding="utf-8"
    )
    print(f"  sha256    : {digest}")


if __name__ == "__main__":
    main()
