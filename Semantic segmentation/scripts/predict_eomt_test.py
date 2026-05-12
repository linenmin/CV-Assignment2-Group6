"""Predict GA2 test masks with a fine-tuned EoMT-DINOv3 checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, EomtDinov3ForUniversalSegmentation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-image-dir", type=Path, default=Path("data/segman_ga2/images/test"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoImageProcessor.from_pretrained(args.checkpoint)
    model = EomtDinov3ForUniversalSegmentation.from_pretrained(args.checkpoint)
    model = model.eval().to(args.device)

    image_paths = sorted(args.test_image_dir.glob("test_*.png"), key=lambda path: int(path.stem.split("_")[1]))
    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    with torch.inference_mode():
        for index, image_path in enumerate(image_paths, start=1):
            image = Image.open(image_path).convert("RGB")
            # Training manually warped images to 512x512 before the HF processor.
            # Reuse the same preprocessing at inference, then map the discrete mask
            # back to the original image size with nearest-neighbor interpolation.
            square_image = image.resize((512, 512), Image.Resampling.BILINEAR)
            inputs = processor(images=square_image, return_tensors="pt").to(args.device)
            outputs = model(**inputs)
            prediction = processor.post_process_semantic_segmentation(
                outputs,
                target_sizes=[(512, 512)],
            )[0]
            prediction_array = prediction.detach().cpu().numpy().astype(np.uint8)
            prediction_array = np.asarray(
                Image.fromarray(prediction_array).resize(image.size, Image.Resampling.NEAREST),
                dtype=np.uint8,
            )
            np.save(args.output_dir / f"{image_path.stem}.npy", prediction_array)

            if index % 50 == 0:
                print(f"predicted={index}", flush=True)

    print(f"prediction_dir={args.output_dir}", flush=True)
    print(f"predictions_written={len(image_paths)}", flush=True)


if __name__ == "__main__":
    main()
