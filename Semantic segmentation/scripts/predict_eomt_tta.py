"""TTA + multi-scale inference for EoMT-DINOv3 on the GA2 split.

The script averages soft semantic probability maps across multiple input
resolutions and an optional horizontal flip. Averaging is done on the
post-Mask2Former semantic score map at a common reference resolution, then
argmax + nearest-resize back to the original image size.

Supports two splits:
  - ``validation``: evaluates against ground-truth masks at the reference
    resolution and writes a per-class IoU summary so the local gain can be
    confirmed before spending a Kaggle submission.
  - ``test``: writes per-image ``.npy`` predictions that match the format
    produced by ``predict_eomt_test.py`` and can be fed directly into
    ``export_submission.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, EomtDinov3ForUniversalSegmentation


IGNORE_INDEX = 255
NUM_CLASSES = 21


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("test", "validation"), default="test")
    parser.add_argument("--data-root", type=Path, default=Path("data/segman_ga2"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--scales",
        type=int,
        nargs="+",
        default=[384, 512, 640],
        help="Square input side lengths used for multi-scale forward passes.",
    )
    parser.add_argument(
        "--reference-size",
        type=int,
        default=512,
        help="Common resolution at which soft semantic maps are averaged.",
    )
    parser.add_argument(
        "--no-hflip",
        action="store_true",
        help="Disable horizontal-flip TTA.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable mixed-precision inference.",
    )
    parser.add_argument("--limit", type=int, default=None)
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
    """Return Mask2Former-style soft semantic score map at ``target_size``.

    Order matches ``EomtImageProcessor.post_process_semantic_segmentation``:
      1. take ``masks_queries_logits`` (raw, low spatial resolution)
      2. bilinear-interpolate to ``target_size``
      3. sigmoid
      4. einsum with softmax over ``class_queries_logits`` (null class dropped)
    Doing the upsample before the sigmoid is important because sigmoid is
    nonlinear; reversing the order shifts the final mIoU by a small but
    non-zero amount.

    ``model.grid_size`` is rewritten per forward so that EoMT's reshape from
    a flat token sequence back into a 2D feature grid works at non-512 input
    sizes. DINOv3's positional embeddings are interpolated dynamically so the
    backbone itself already handles variable input.
    """

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
    scales: Iterable[int],
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
    autocast_kwargs = {"device_type": "cuda", "enabled": use_amp and device.type == "cuda"}

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


def iter_validation_samples(data_root: Path) -> list[tuple[str, Path, Path]]:
    split_file = data_root / "validation.txt"
    image_dir = data_root / "images" / "validation"
    mask_dir = data_root / "annotations" / "validation"
    samples: list[tuple[str, Path, Path]] = []
    for line in split_file.read_text(encoding="utf-8").splitlines():
        stem = line.strip()
        if not stem:
            continue
        samples.append((stem, image_dir / f"{stem}.png", mask_dir / f"{stem}.png"))
    return samples


def iter_test_samples(data_root: Path) -> list[tuple[str, Path, None]]:
    image_dir = data_root / "images" / "test"
    paths = sorted(image_dir.glob("test_*.png"), key=lambda p: int(p.stem.split("_")[1]))
    return [(p.stem, p, None) for p in paths]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    use_hflip = not args.no_hflip
    use_amp = not args.no_amp

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
                f"Scale {scale} is not a multiple of patch_size={patch_size}; "
                "EoMT requires patch-aligned inputs."
            )

    print(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "split": args.split,
                "scales": list(args.scales),
                "reference_size": args.reference_size,
                "hflip": use_hflip,
                "amp": use_amp,
                "image_mean": mean.tolist(),
                "image_std": std.tolist(),
            },
            indent=2,
        ),
        flush=True,
    )

    if args.split == "test":
        samples = iter_test_samples(args.data_root)
    else:
        samples = iter_validation_samples(args.data_root)
    if args.limit is not None:
        samples = samples[: args.limit]

    inter_primary = np.zeros(NUM_CLASSES, dtype=np.float64)
    union_primary = np.zeros(NUM_CLASSES, dtype=np.float64)
    inter_original = np.zeros(NUM_CLASSES, dtype=np.float64)
    union_original = np.zeros(NUM_CLASSES, dtype=np.float64)

    with torch.inference_mode():
        for index, (stem, image_path, mask_path) in enumerate(samples, start=1):
            image = Image.open(image_path).convert("RGB")
            pred_ref = tta_predict(
                model=model,
                pil_image=image,
                scales=args.scales,
                reference_size=args.reference_size,
                use_hflip=use_hflip,
                mean=mean,
                std=std,
                device=device,
                use_amp=use_amp,
                patch_size=patch_size,
            )

            if args.split == "test":
                pred_full = np.asarray(
                    Image.fromarray(pred_ref).resize(image.size, Image.Resampling.NEAREST),
                    dtype=np.uint8,
                )
                np.save(args.output_dir / f"{stem}.npy", pred_full)
            else:
                mask_pil = Image.open(mask_path)
                target_primary = np.asarray(
                    mask_pil.resize(
                        (args.reference_size, args.reference_size),
                        Image.Resampling.NEAREST,
                    ),
                    dtype=np.int64,
                )
                target_original = np.asarray(mask_pil, dtype=np.int64)
                pred_original = np.asarray(
                    Image.fromarray(pred_ref).resize(image.size, Image.Resampling.NEAREST),
                    dtype=np.uint8,
                )
                for class_id in range(NUM_CLASSES):
                    valid_p = target_primary != IGNORE_INDEX
                    pm_p = (pred_ref == class_id) & valid_p
                    tm_p = (target_primary == class_id) & valid_p
                    inter_primary[class_id] += np.logical_and(pm_p, tm_p).sum()
                    union_primary[class_id] += np.logical_or(pm_p, tm_p).sum()

                    valid_o = target_original != IGNORE_INDEX
                    pm_o = (pred_original == class_id) & valid_o
                    tm_o = (target_original == class_id) & valid_o
                    inter_original[class_id] += np.logical_and(pm_o, tm_o).sum()
                    union_original[class_id] += np.logical_or(pm_o, tm_o).sum()

            if index % 25 == 0:
                print(f"processed={index}/{len(samples)}", flush=True)

    if args.split == "validation":
        def _finalize(inter: np.ndarray, union: np.ndarray) -> tuple[float, dict[int, float]]:
            per_class = {
                int(class_id): float(inter[class_id] / union[class_id])
                for class_id in range(NUM_CLASSES)
                if union[class_id] > 0
            }
            miou = float(np.mean(list(per_class.values()))) if per_class else 0.0
            return miou, per_class

        miou_primary, per_class_primary = _finalize(inter_primary, union_primary)
        miou_original, per_class_original = _finalize(inter_original, union_original)
        summary = {
            "checkpoint": str(args.checkpoint),
            "scales": list(args.scales),
            "reference_size": args.reference_size,
            "hflip": use_hflip,
            "amp": use_amp,
            "num_samples": len(samples),
            "mIoU_primary_512": miou_primary,
            "mIoU_original_size": miou_original,
            "per_class_iou_primary_512": per_class_primary,
            "per_class_iou_original_size": per_class_original,
        }
        out_path = args.output_dir / "tta_validation_summary.json"
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"validation_mIoU_primary_512={miou_primary:.4f}", flush=True)
        print(f"validation_mIoU_original_size={miou_original:.4f}", flush=True)
        for class_id in sorted(per_class_primary):
            print(
                f"class={class_id:02d} iou_512={per_class_primary[class_id]:.4f} "
                f"iou_orig={per_class_original.get(class_id, float('nan')):.4f}",
                flush=True,
            )
        print(f"summary_path={out_path}", flush=True)

    print(f"output_dir={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
