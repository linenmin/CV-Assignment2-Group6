"""Smoke test for loading and running the EoMT-DINOv3 ADE20K checkpoint."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, EomtDinov3ForUniversalSegmentation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        default="tue-mps/eomt-dinov3-ade-semantic-large-512",
        help="Hugging Face model id or local checkpoint directory.",
    )
    parser.add_argument(
        "--image",
        default="data/segman_ga2/images/validation/train_10.png",
        help="Image path used for the one-image smoke test.",
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Smoke-test image does not exist: {image_path}")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")

    print(f"model_id={args.model_id}", flush=True)
    print(f"image={image_path}", flush=True)
    print(f"device={args.device}", flush=True)

    processor = AutoImageProcessor.from_pretrained(args.model_id)
    model = EomtDinov3ForUniversalSegmentation.from_pretrained(args.model_id, dtype="auto")
    model = model.eval().to(args.device)

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(args.device)
    print(f"input_size={image.size}", flush=True)
    print(f"pixel_values_shape={tuple(inputs['pixel_values'].shape)}", flush=True)
    print(f"pixel_values_dtype={inputs['pixel_values'].dtype}", flush=True)

    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    with torch.inference_mode():
        outputs = model(**inputs)
    forward_sec = time.perf_counter() - start

    result = processor.post_process_semantic_segmentation(
        outputs,
        target_sizes=[image.size[::-1]],
    )[0]

    print(f"forward_sec={forward_sec:.3f}", flush=True)
    print(f"result_shape={tuple(result.shape)}", flush=True)
    print(f"result_dtype={result.dtype}", flush=True)
    print(f"result_label_min={int(result.min())}", flush=True)
    print(f"result_label_max={int(result.max())}", flush=True)

    if args.device.startswith("cuda"):
        peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"peak_memory_mb={peak_mb:.1f}", flush=True)


if __name__ == "__main__":
    main()
