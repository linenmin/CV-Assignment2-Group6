"""Recompute a Cityscapes external-generalization summary from an existing
``metrics.csv`` so that the primary mIoU excludes the ``train`` class.

Why: ``train`` appears in both VOC and Cityscapes label sets only by name.
Visual inspection (see ``outputs/figures/train_class_voc_vs_cityscapes/``)
shows VOC trains are full-frame intercity / steam locomotives and
Cityscapes "trains" are urban trams; they are essentially different
visual concepts and including ``train`` deflates the overlap mIoU with a
quantity that is not really measuring transfer at all.

This script does NOT re-run inference. It reads the per-class IoU rows
already produced by the SegMAN or EoMT evaluator and emits a refreshed
``summary.md`` that reports both the 5-class transferable mIoU
(``person, car, bus, motorbike, bicycle``) as primary and the original
6-class number as secondary, for transparency.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ga2_seg.cityscapes_generalization import (  # noqa: E402
    CITYSCAPES_LABEL_TO_VOC,
    OVERLAP_CLASS_NAMES,
    TRANSFERABLE_CLASS_NAMES,
)
from ga2_seg.labels import CLASS_NAME_TO_LABEL_ID  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument(
        "--summary-md",
        type=Path,
        default=None,
        help="Output path. Defaults to metrics.csv's sibling summary.md.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="",
        help="Optional label prefix for the heading (e.g. 'EoMT-DINOv3 V17').",
    )
    parser.add_argument(
        "--source-info",
        type=str,
        default="",
        help="Optional one-line metadata to embed (checkpoint, inference, roots).",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean_iou(rows: list[dict[str, str]], class_names: tuple[str, ...]) -> float:
    keep = [r for r in rows if r["class_name"] in class_names]
    ious = [float(r["iou"]) for r in keep if r["iou"] not in ("", "nan")]
    return float(np.mean(ious)) if ious else float("nan")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.metrics_csv)
    out_path = args.summary_md or args.metrics_csv.with_name("summary.md")

    miou_5 = mean_iou(rows, TRANSFERABLE_CLASS_NAMES)
    miou_6 = mean_iou(rows, OVERLAP_CLASS_NAMES)

    heading_label = f" ({args.label})" if args.label else ""
    lines = [
        f"# Cityscapes External Generalization Summary{heading_label}",
        "",
    ]
    if args.source_info:
        lines.append(f"- Source: `{args.source_info}`")
    lines.extend(
        [
            f"- Primary metric: 5-class transferable mIoU (excludes `train`):"
            f" **`{miou_5:.4f}`**",
            f"- Secondary: 6-class overlap mIoU (includes `train`, kept for backward"
            f" comparison): `{miou_6:.4f}`",
            "",
            "`train` is excluded from the primary metric because VOC trains and",
            "Cityscapes 'trains' are essentially different visual concepts: VOC trains",
            "are full-frame intercity / steam locomotives photographed as subject,",
            "while Cityscapes 'trains' are urban trams that appear small and",
            "incidentally in street scenes. See",
            "`outputs/figures/train_class_voc_vs_cityscapes/` for visual evidence.",
            "",
            "## Class Mapping",
            "",
        ]
    )
    for cityscapes_id, voc_id in sorted(CITYSCAPES_LABEL_TO_VOC.items()):
        class_name = next(
            name for name, label_id in CLASS_NAME_TO_LABEL_ID.items() if label_id == voc_id
        )
        suffix = "" if class_name in TRANSFERABLE_CLASS_NAMES else "  (excluded from primary metric)"
        lines.append(
            f"- Cityscapes `{cityscapes_id}` -> VOC `{class_name}` (`{voc_id}`){suffix}"
        )

    lines.extend(["", "## Per-Class IoU", ""])
    for row in rows:
        iou = row["iou"]
        try:
            iou_text = "nan" if iou in ("", "nan") else f"{float(iou):.4f}"
        except ValueError:
            iou_text = "nan"
        flag = "" if row["class_name"] in TRANSFERABLE_CLASS_NAMES else "  (excluded)"
        lines.append(
            f"- `{row['class_name']}`: IoU `{iou_text}`, "
            f"target pixels `{row['target_pixels']}`, union `{row['union']}`{flag}"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"summary_path={out_path}")
    print(f"primary_5class_miou={miou_5:.4f}")
    print(f"secondary_6class_miou={miou_6:.4f}")


if __name__ == "__main__":
    main()
