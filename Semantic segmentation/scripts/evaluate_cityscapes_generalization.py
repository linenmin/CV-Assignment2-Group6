import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ga2_seg.cityscapes_generalization import (
    CITYSCAPES_LABEL_TO_VOC,
    IGNORE_LABEL,
    OVERLAP_CLASS_NAMES,
    collect_cityscapes_pairs,
    colorize_overlap_mask,
    compute_overlap_intersections_unions,
    map_cityscapes_label_ids,
    resize_prediction_to_label,
    resolve_cityscapes_roots,
    summarize_iou,
)
from ga2_seg.labels import CLASS_NAME_TO_LABEL_ID
from ga2_seg.paths import get_project_root


def parse_args() -> argparse.Namespace:
    project_root = get_project_root()
    parser = argparse.ArgumentParser(description="Evaluate SegMAN-B on Cityscapes overlap classes.")
    parser.add_argument("--cityscapes-root", type=Path, default=project_root / "data" / "cityscapes")
    parser.add_argument("--left-img-root", type=Path, default=None)
    parser.add_argument("--gt-fine-root", type=Path, default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "external" / "SegMAN" / "segmentation" / "local_configs" / "segman" / "ga2" / "segman_b_ga2.py",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=project_root
        / "external"
        / "SegMAN"
        / "segmentation"
        / "outputs"
        / "ga2_segman_b"
        / "best_mIoU_iter_25000.pth",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "outputs" / "cityscapes_generalization" / "segman_b_iter25000",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-visualizations", type=int, default=12)
    return parser.parse_args()


def _write_metrics_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["class_id", "class_name", "intersection", "union", "target_pixels", "iou"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(
    path: Path,
    rows: list[dict[str, object]],
    num_samples: int,
    roots_text: str,
) -> None:
    valid_ious = [row["iou"] for row in rows if not np.isnan(float(row["iou"]))]
    overlap_miou = float(np.mean(valid_ious)) if valid_ious else float("nan")
    lines = [
        "# Cityscapes External Generalization Summary",
        "",
        f"- Samples evaluated: `{num_samples}`",
        f"- Dataset roots: `{roots_text}`",
        f"- Metric: `overlap-class mIoU`",
        f"- Overlap mIoU: `{overlap_miou:.4f}`",
        "",
        "This is not a full VOC-20 evaluation. Non-overlap Cityscapes classes are ignored.",
        "",
        "## Class Mapping",
        "",
    ]
    for cityscapes_id, voc_id in sorted(CITYSCAPES_LABEL_TO_VOC.items()):
        class_name = next(name for name, label_id in CLASS_NAME_TO_LABEL_ID.items() if label_id == voc_id)
        lines.append(f"- Cityscapes `{cityscapes_id}` -> VOC `{class_name}` (`{voc_id}`)")

    lines.extend(["", "## Per-Class IoU", ""])
    for row in rows:
        iou = float(row["iou"])
        iou_text = "nan" if np.isnan(iou) else f"{iou:.4f}"
        lines.append(
            f"- `{row['class_name']}`: IoU `{iou_text}`, target pixels `{row['target_pixels']}`, union `{row['union']}`"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_visualization(path: Path, image: np.ndarray, target: np.ndarray, prediction: np.ndarray) -> None:
    target_vis = colorize_overlap_mask(target)
    pred_vis = colorize_overlap_mask(prediction)
    valid = target != IGNORE_LABEL
    error = np.zeros_like(image)
    error[np.logical_and(valid, target == prediction)] = (40, 180, 80)
    error[np.logical_and(valid, target != prediction)] = (220, 60, 50)

    panels = [image, target_vis, pred_vis, error]
    canvas = np.concatenate(panels, axis=1)
    Image.fromarray(canvas).save(path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = args.output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    roots = resolve_cityscapes_roots(
        cityscapes_root=args.cityscapes_root,
        left_img_root=args.left_img_root,
        gt_fine_root=args.gt_fine_root,
        split=args.split,
    )
    pairs = collect_cityscapes_pairs(roots, limit=args.limit)
    from mmseg.apis import inference_segmentor, init_segmentor

    model = init_segmentor(str(args.config), str(args.checkpoint), device=args.device)

    class_ids = tuple(CLASS_NAME_TO_LABEL_ID[name] for name in OVERLAP_CLASS_NAMES)
    intersections = {class_id: 0 for class_id in class_ids}
    unions = {class_id: 0 for class_id in class_ids}
    target_pixels = {class_id: 0 for class_id in class_ids}

    for index, pair in enumerate(pairs, start=1):
        result = inference_segmentor(model, str(pair.image_path))
        prediction = np.asarray(result[0], dtype=np.uint8)
        label_ids = np.asarray(Image.open(pair.label_path), dtype=np.uint8)
        target = map_cityscapes_label_ids(label_ids)
        prediction = resize_prediction_to_label(prediction, target.shape)

        sample_intersections, sample_unions, sample_target_pixels = compute_overlap_intersections_unions(
            prediction=prediction,
            target=target,
            class_ids=class_ids,
        )
        for class_id in class_ids:
            intersections[class_id] += sample_intersections[class_id]
            unions[class_id] += sample_unions[class_id]
            target_pixels[class_id] += sample_target_pixels[class_id]

        if index <= args.num_visualizations:
            image = np.asarray(Image.open(pair.image_path).convert("RGB"), dtype=np.uint8)
            _save_visualization(vis_dir / f"{pair.sample_id}.png", image, target, prediction)

        if index % 25 == 0:
            print(f"evaluated={index}")

    rows = summarize_iou(intersections, unions, target_pixels)
    _write_metrics_csv(args.output_dir / "metrics.csv", rows)
    _write_summary(
        args.output_dir / "summary.md",
        rows,
        num_samples=len(pairs),
        roots_text=f"leftImg8bit={roots.left_img_root}; gtFine={roots.gt_fine_root}; split={roots.split}",
    )
    print(f"output_dir={args.output_dir}")
    print(f"samples_evaluated={len(pairs)}")


if __name__ == "__main__":
    main()
