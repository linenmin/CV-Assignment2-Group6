"""Evaluate an EoMT-DINOv3 checkpoint on Cityscapes overlap classes.

Mirrors ``scripts/evaluate_cityscapes_generalization.py`` (SegMAN/mmseg)
but plugs in the V16/V17 EoMT-DINOv3 inference path:

  1. resize image to a square of ``window_size`` (default 512x512) with
     bilinear interpolation, matching the V11.. fine-tune preprocessing;
  2. run the model at each requested input scale, dynamically rewriting
     ``model.grid_size`` to honour the patch-aligned grid assumption;
  3. average Mask2Former-style soft semantic scores at a common 512x512
     reference and argmax;
  4. nearest-resize the discrete mask back to the original Cityscapes
     image resolution before computing per-class IoU.

Only the six VOC categories that overlap Cityscapes are scored
(``person, car, bus, train, motorbike, bicycle``); all other Cityscapes
labels collapse to ``IGNORE_LABEL`` so the metric reflects realistic
domain transfer rather than full VOC-20 accuracy.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, EomtDinov3ForUniversalSegmentation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ga2_seg.cityscapes_generalization import (  # noqa: E402
    CITYSCAPES_LABEL_TO_VOC,
    IGNORE_LABEL,
    OVERLAP_CLASS_NAMES,
    collect_cityscapes_pairs,
    colorize_overlap_mask,
    compute_overlap_intersections_unions,
    map_cityscapes_label_ids,
    resolve_cityscapes_roots,
    summarize_iou,
)
from ga2_seg.labels import CLASS_NAME_TO_LABEL_ID  # noqa: E402
from ga2_seg.paths import get_project_root  # noqa: E402


NUM_CLASSES = 21


def parse_args() -> argparse.Namespace:
    project_root = get_project_root()
    parser = argparse.ArgumentParser(
        description="Evaluate EoMT-DINOv3 V17 on Cityscapes overlap classes."
    )
    parser.add_argument(
        "--cityscapes-root", type=Path, default=project_root / "data" / "cityscapes"
    )
    parser.add_argument("--left-img-root", type=Path, default=None)
    parser.add_argument("--gt-fine-root", type=Path, default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=project_root
        / "outputs"
        / "checkpoints"
        / "eomt_dinov3_v17_merge_val_lr1e5"
        / "best",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root
        / "outputs"
        / "cityscapes_generalization"
        / "eomt_dinov3_v17_tta_ms496_512_528_noflip",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-visualizations", type=int, default=12)
    parser.add_argument(
        "--scales",
        type=int,
        nargs="+",
        default=[496, 512, 528],
        help="Input side lengths used for multi-scale TTA averaging.",
    )
    parser.add_argument("--reference-size", type=int, default=512)
    parser.add_argument(
        "--use-hflip",
        action="store_true",
        help="Add horizontal-flip TTA on top of multi-scale (off by default; "
        "hflip regressed on the GA2 validation set).",
    )
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def normalize_to_tensor(
    pil_image: Image.Image,
    scale: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> torch.Tensor:
    image = pil_image.resize((scale, scale), Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - mean) / std
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous().float()
    return tensor


def compute_semantic_score(
    model: EomtDinov3ForUniversalSegmentation,
    pixel_values: torch.Tensor,
    target_size: tuple[int, int],
    patch_size: int,
) -> torch.Tensor:
    height, width = pixel_values.shape[-2:]
    model.grid_size = (height // patch_size, width // patch_size)
    outputs = model(pixel_values=pixel_values)
    mask_logits = outputs.masks_queries_logits.float()
    class_logits = outputs.class_queries_logits.float()
    mask_logits = F.interpolate(mask_logits, size=target_size, mode="bilinear")
    mask_probs = mask_logits.sigmoid()
    class_probs = class_logits.softmax(dim=-1)[..., :-1]
    semantic = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)
    return semantic


def tta_predict(
    model: EomtDinov3ForUniversalSegmentation,
    pil_image: Image.Image,
    scales: list[int],
    reference_size: int,
    use_hflip: bool,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    use_amp: bool,
    patch_size: int,
) -> np.ndarray:
    target = (reference_size, reference_size)
    accum: torch.Tensor | None = None
    count = 0
    autocast_kwargs = {
        "device_type": "cuda",
        "enabled": use_amp and device.type == "cuda",
    }

    for scale in scales:
        tensor = normalize_to_tensor(pil_image, scale, mean, std).to(device)
        with torch.amp.autocast(**autocast_kwargs):
            score = compute_semantic_score(model, tensor, target, patch_size)
        score = score.float()
        accum = score if accum is None else accum + score
        count += 1

        if use_hflip:
            flipped = pil_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            tensor_f = normalize_to_tensor(flipped, scale, mean, std).to(device)
            with torch.amp.autocast(**autocast_kwargs):
                score_f = compute_semantic_score(model, tensor_f, target, patch_size)
            score_f = torch.flip(score_f.float(), dims=[-1])
            accum = accum + score_f
            count += 1

    assert accum is not None
    accum = accum / count
    pred = accum.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
    return pred


def resize_mask_nearest(mask: np.ndarray, target_size_wh: tuple[int, int]) -> np.ndarray:
    return np.asarray(
        Image.fromarray(mask).resize(target_size_wh, Image.Resampling.NEAREST),
        dtype=np.uint8,
    )


def _write_metrics_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "class_id",
                "class_name",
                "intersection",
                "union",
                "target_pixels",
                "iou",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(
    path: Path,
    rows: list[dict[str, object]],
    num_samples: int,
    roots_text: str,
    checkpoint_text: str,
    inference_text: str,
) -> None:
    valid_ious = [row["iou"] for row in rows if not np.isnan(float(row["iou"]))]
    overlap_miou = float(np.mean(valid_ious)) if valid_ious else float("nan")
    lines = [
        "# Cityscapes External Generalization Summary (EoMT-DINOv3 V17)",
        "",
        f"- Samples evaluated: `{num_samples}`",
        f"- Dataset roots: `{roots_text}`",
        f"- Checkpoint: `{checkpoint_text}`",
        f"- Inference: `{inference_text}`",
        f"- Metric: `overlap-class mIoU`",
        f"- Overlap mIoU: `{overlap_miou:.4f}`",
        "",
        "This is not a full VOC-20 evaluation. Non-overlap Cityscapes classes are ignored.",
        "",
        "## Class Mapping",
        "",
    ]
    for cityscapes_id, voc_id in sorted(CITYSCAPES_LABEL_TO_VOC.items()):
        class_name = next(
            name for name, label_id in CLASS_NAME_TO_LABEL_ID.items() if label_id == voc_id
        )
        lines.append(
            f"- Cityscapes `{cityscapes_id}` -> VOC `{class_name}` (`{voc_id}`)"
        )

    lines.extend(["", "## Per-Class IoU", ""])
    for row in rows:
        iou = float(row["iou"])
        iou_text = "nan" if np.isnan(iou) else f"{iou:.4f}"
        lines.append(
            f"- `{row['class_name']}`: IoU `{iou_text}`, "
            f"target pixels `{row['target_pixels']}`, union `{row['union']}`"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_visualization(
    path: Path, image: np.ndarray, target: np.ndarray, prediction: np.ndarray
) -> None:
    target_vis = colorize_overlap_mask(target)
    pred_vis = colorize_overlap_mask(prediction)
    valid = target != IGNORE_LABEL
    error = np.zeros_like(image)
    error[np.logical_and(valid, target == prediction)] = (40, 180, 80)
    error[np.logical_and(valid, target != prediction)] = (220, 60, 50)
    canvas = np.concatenate([image, target_vis, pred_vis, error], axis=1)
    Image.fromarray(canvas).save(path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = args.output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    use_amp = not args.no_amp

    roots = resolve_cityscapes_roots(
        cityscapes_root=args.cityscapes_root,
        left_img_root=args.left_img_root,
        gt_fine_root=args.gt_fine_root,
        split=args.split,
    )
    pairs = collect_cityscapes_pairs(roots, limit=args.limit)

    processor = AutoImageProcessor.from_pretrained(args.checkpoint)
    mean = np.asarray(processor.image_mean, dtype=np.float32)
    std = np.asarray(processor.image_std, dtype=np.float32)
    model = (
        EomtDinov3ForUniversalSegmentation.from_pretrained(args.checkpoint)
        .eval()
        .to(device)
    )
    patch_size = int(model.config.patch_size)
    for scale in args.scales:
        if scale % patch_size != 0:
            raise ValueError(
                f"Scale {scale} is not a multiple of patch_size={patch_size}."
            )

    class_ids = tuple(CLASS_NAME_TO_LABEL_ID[name] for name in OVERLAP_CLASS_NAMES)
    intersections = {cid: 0 for cid in class_ids}
    unions = {cid: 0 for cid in class_ids}
    target_pixels = {cid: 0 for cid in class_ids}

    inference_text = (
        f"scales={list(args.scales)} ref={args.reference_size} hflip={args.use_hflip}"
    )
    print(
        f"checkpoint={args.checkpoint}\n"
        f"split={args.split} samples={len(pairs)} inference={inference_text}",
        flush=True,
    )

    with torch.inference_mode():
        for index, pair in enumerate(pairs, start=1):
            image_pil = Image.open(pair.image_path).convert("RGB")
            pred_ref = tta_predict(
                model=model,
                pil_image=image_pil,
                scales=args.scales,
                reference_size=args.reference_size,
                use_hflip=args.use_hflip,
                mean=mean,
                std=std,
                device=device,
                use_amp=use_amp,
                patch_size=patch_size,
            )

            label_ids = np.asarray(Image.open(pair.label_path), dtype=np.uint8)
            target = map_cityscapes_label_ids(label_ids)
            prediction = resize_mask_nearest(pred_ref, (target.shape[1], target.shape[0]))

            sample_intersections, sample_unions, sample_target_pixels = (
                compute_overlap_intersections_unions(
                    prediction=prediction,
                    target=target,
                    class_ids=class_ids,
                )
            )
            for cid in class_ids:
                intersections[cid] += sample_intersections[cid]
                unions[cid] += sample_unions[cid]
                target_pixels[cid] += sample_target_pixels[cid]

            if index <= args.num_visualizations:
                image_arr = np.asarray(image_pil, dtype=np.uint8)
                _save_visualization(
                    vis_dir / f"{pair.sample_id}.png", image_arr, target, prediction
                )

            if index % 25 == 0:
                print(f"evaluated={index}/{len(pairs)}", flush=True)

    rows = summarize_iou(intersections, unions, target_pixels)
    _write_metrics_csv(args.output_dir / "metrics.csv", rows)
    _write_summary(
        args.output_dir / "summary.md",
        rows,
        num_samples=len(pairs),
        roots_text=(
            f"leftImg8bit={roots.left_img_root}; "
            f"gtFine={roots.gt_fine_root}; split={roots.split}"
        ),
        checkpoint_text=str(args.checkpoint),
        inference_text=inference_text,
    )
    print(f"output_dir={args.output_dir}")
    print(f"samples_evaluated={len(pairs)}")
    valid_ious = [row["iou"] for row in rows if not np.isnan(float(row["iou"]))]
    if valid_ious:
        print(f"overlap_miou={float(np.mean(valid_ious)):.4f}")
    for row in rows:
        iou = float(row["iou"])
        iou_text = "nan" if np.isnan(iou) else f"{iou:.4f}"
        print(f"class={row['class_name']:<10} iou={iou_text}")


if __name__ == "__main__":
    main()
