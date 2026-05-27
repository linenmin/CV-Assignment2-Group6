"""Audit which dataset files are actually referenced by the notebook
and report any that are not used (candidates for removal)."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

ROOT = Path(
    r"D:\BaiduNetdiskWorkspace\Leuven\8th\Computer Vision\assignment\Group2\kaggle_submission"
)
NB_PATH = ROOT / "notebook" / "group6_final.ipynb"
DATASET = ROOT / "dataset"


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    referenced: set[str] = set()
    for c in nb["cells"]:
        src = "".join(c.get("source", []))
        for m in re.findall(r'"([^"]*\.(?:png|svg|csv|npy|pth|safetensors|json))"', src):
            base = m.replace("\\", "/").rsplit("/", 1)[-1]
            referenced.add(base)

    all_files: dict[str, int] = {}
    for dirpath, _, names in os.walk(DATASET):
        for n in names:
            p = Path(dirpath) / n
            rel = p.relative_to(DATASET).as_posix()
            all_files[rel] = p.stat().st_size

    print("=== Files NOT referenced in notebook source ===")
    unused = []
    for rel, size in sorted(all_files.items()):
        if "__pycache__" in rel:
            unused.append((rel, size, "pycache"))
            continue
        base = rel.rsplit("/", 1)[-1]
        if base not in referenced:
            unused.append((rel, size, "unreferenced"))
    total_unused = 0
    for rel, size, why in sorted(unused):
        mb = size / 1024 / 1024
        print(f"  [{why}] {rel}  ({mb:.2f} MB)")
        total_unused += size
    print(f"\nTotal unused: {total_unused/1024/1024:.1f} MB out of "
          f"{sum(all_files.values())/1024/1024:.1f} MB total")

    print()
    print("=== File extension breakdown ===")
    by_ext: dict[str, list[int]] = {}
    for rel, size in all_files.items():
        if "__pycache__" in rel:
            continue
        ext = "." + rel.rsplit(".", 1)[-1] if "." in rel else "(noext)"
        by_ext.setdefault(ext, []).append(size)
    for ext, sizes in sorted(by_ext.items(), key=lambda kv: -sum(kv[1])):
        print(f"  {ext:18s} count={len(sizes):3d}  total={sum(sizes)/1024/1024:.1f} MB")


if __name__ == "__main__":
    main()
