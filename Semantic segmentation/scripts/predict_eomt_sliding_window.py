"""Sliding-window inference for EoMT-DINOv3.

Resize the input so its shorter side equals the trained window size while
keeping the aspect ratio, then run overlapping square windows over the
result. Each window is fed at exactly the trained ``512x512`` resolution,
so EoMT's hard-coded ``32x32`` token grid is honoured for free without
monkey-patching ``model.grid_size`` (unlike the multi-scale TTA path).

Mask2Former-style soft semantic score maps are accumulated in image space
with a per-pixel count buffer, then averaged in overlapping regions,
argmaxed, and nearest-resized back to the original image size.

Two evaluation modes are reported when ``--split validation`` is set:
  - ``primary`` mIoU at ``512x512`` warped resolution, matching the
    protocol used by the V11..V16 fine-tune logs and ``predict_eomt_tta``.
  - ``original`` mIoU at each image's native resolution, which is closer
    to what Kaggle actually scores.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--use-hflip", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def resize_keep_aspect(pil: Image.Image, short_side: int) -> Image.Image:
    w, h = pil.size
    if min(w, h) == short_side:
        return pil
    scale = short_side / float(min(w, h))
    new_w = max(short_side, int(round(w * scale)))
    new_h = max(short_side, int(round(h * scale)))
    return pil.resize((new_w, new_h), Image.Resampling.BILINEAR)


def window_starts(length: int, window: int, stride: int) -> list[int]:
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, stride))
    if starts[-1] + window < length:
        starts.append(length - window)
    return starts


def normalize_array(arr_uint8: np.ndarray, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    arr = arr_uint8.astype(np.float32) / 255.0
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


def sliding_window_predict(
    model: EomtDinov3ForUniversalSegmentation,
    pil_image: Image.Image,
    window: int,
    stride: int,
    use_hflip: bool,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    use_amp: bool,
    patch_size: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Return ``(pred, (H_resized, W_resized))`` at the aspect-preserved size."""

    resized = resize_keep_aspect(pil_image, window)
    W, H = resized.size
    arr = np.asarray(resized, dtype=np.uint8)

    accum = torch.zeros(1, NUM_CLASSES, H, W, dtype=torch.float32, device=device)
    counts = torch.zeros(1, 1, H, W, dtype=torch.float32, device=device)
    autocast_kwargs = {"device_type": "cuda", "enabled": use_amp and device.type == "cuda"}

    ys = window_starts(H, window, stride)
    xs = window_starts(W, window, stride)

    for y0 in ys:
        for x0 in xs:
            crop = arr[y0 : y0 + window, x0 : x0 + window, :]
            tensor = normalize_array(crop, mean, std).to(device)
            with torch.amp.autocast(**autocast_kwargs):
                score = compute_semantic_score(model, tensor, (window, window), patch_size)
            accum[:, :, y0 : y0 + window, x0 : x0 + window] += score.float()
            counts[:, :, y0 : y0 + window, x0 : x0 + window] += 1.0

            if use_hflip:
                crop_f = np.ascontiguousarray(crop[:, ::-1, :])
                tensor_f = normalize_array(crop_f, mean, std).to(device)
                with torch.amp.autocast(**autocast_kwargs):
                    score_f = compute_semantic_score(model, tensor_f, (window, window), patch_size)
                score_f = torch.flip(score_f.float(), dims=[-1])
                accum[:, :, y0 : y0 + window, x0 : x0 + window] += score_f
                counts[:, :, y0 : y0 + window, x0 : x0 + window] += 1.0

    semantic = accum / counts.clamp(min=1.0)
    pred = semantic.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
    return pred, (H, W)


def resize_mask(pred: np.ndarray, target_size_wh: tuple[int, int]) -> np.ndarray:
    return np.asarray(
        Image.fromarray(pred).resize(target_size_wh, Image.Resampling.NEAREST),
        dtype=np.uint8,
    )


def update_iou_buffers(
    pred: np.ndarray,
    target: np.ndarray,
    intersections: np.ndarray,
    unions: np.ndarray,
) -> None:
    valid = target != IGNORE_INDEX
    for class_id in range(NUM_CLASSES):
        pm = (pred == class_id) & valid
        tm = (target == class_id) & valid
        intersections[class_id] += np.logical_and(pm, tm).sum()
        unions[class_id] += np.logical_or(pm, tm).sum()


def finalize_per_class(
    intersections: np.ndarray, unions: np.ndarray
) -> tuple[float, dict[int, float]]:
    per_class = {
        int(class_id): float(intersections[class_id] / unions[class_id])
        for class_id in range(NUM_CLASSES)
        if unions[class_id] > 0
    }
    miou = float(np.mean(list(per_class.values()))) if per_class else 0.0
    return miou, per_class


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
    use_amp = not args.no_amp

    if args.window_size % 16 != 0:
        raise ValueError(f"window_size={args.window_size} must be a multiple of 16.")
    if args.stride <= 0 or args.stride > args.window_size:
        raise ValueError(
            f"stride={args.stride} must be in (0, window_size={args.window_size}]."
        )

    processor = AutoImageProcessor.from_pretrained(args.checkpoint)
    mean = np.asarray(processor.image_mean, dtype=np.float32)
    std = np.asarray(processor.image_std, dtype=np.float32)
    model = (
        EomtDinov3ForUniversalSegmentation.from_pretrained(args.checkpoint)
        .eval()
        .to(device)
    )
    patch_size = int(model.config.patch_size)

    print(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "split": args.split,
                "window_size": args.window_size,
                "stride": args.stride,
                "use_hflip": args.use_hflip,
                "use_amp": use_amp,
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
            original_w, original_h = image.size
            pred_resized, (H_r, W_r) = sliding_window_predict(
                model=model,
                pil_image=image,
                window=args.window_size,
                stride=args.stride,
                use_hflip=args.use_hflip,
                mean=mean,
                std=std,
                device=device,
                use_amp=use_amp,
                patch_size=patch_size,
            )

            pred_original = resize_mask(pred_resized, (original_w, original_h))

            if args.split == "test":
                np.save(args.output_dir / f"{stem}.npy", pred_original)
            else:
                mask_pil = Image.open(mask_path)
                pred_primary = resize_mask(pred_resized, (args.window_size, args.window_size))
                target_primary = np.asarray(
                    mask_pil.resize(
                        (args.window_size, args.window_size), Image.Resampling.NEAREST
                    ),
                    dtype=np.int64,
                )
                target_original = np.asarray(mask_pil, dtype=np.int64)
                update_iou_buffers(pred_primary, target_primary, inter_primary, union_primary)
                update_iou_buffers(
                    pred_original, target_original, inter_original, union_original
                )

            if index % 25 == 0:
                print(f"processed={index}/{len(samples)}", flush=True)

    if args.split == "validation":
        miou_primary, per_class_primary = finalize_per_class(inter_primary, union_primary)
        miou_original, per_class_original = finalize_per_class(
            inter_original, union_original
        )
        summary = {
            "checkpoint": str(args.checkpoint),
            "window_size": args.window_size,
            "stride": args.stride,
            "use_hflip": args.use_hflip,
            "use_amp": use_amp,
            "num_samples": len(samples),
            "mIoU_primary_512": miou_primary,
            "mIoU_original_size": miou_original,
            "per_class_iou_primary_512": per_class_primary,
            "per_class_iou_original_size": per_class_original,
        }
        out_path = args.output_dir / "sliding_window_validation_summary.json"
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
