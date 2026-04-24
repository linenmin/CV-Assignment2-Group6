import argparse
from pathlib import Path

import numpy as np
from mmseg.apis import inference_segmentor, init_segmentor

from ga2_seg.segman_adapter import get_segman_export_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict GA2 test masks with an official SegMAN checkpoint.")
    parser.add_argument("--config", type=Path, required=True, help="SegMAN GA2 config path.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trained SegMAN checkpoint path.")
    parser.add_argument(
        "--test-image-dir",
        type=Path,
        default=get_segman_export_root() / "images" / "test",
        help="Directory containing exported test_*.png files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where test_<id>.npy masks will be written.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = init_segmentor(str(args.config), str(args.checkpoint), device=args.device)
    image_paths = sorted(args.test_image_dir.glob("test_*.png"), key=lambda path: int(path.stem.split("_")[1]))
    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    for index, image_path in enumerate(image_paths, start=1):
        result = inference_segmentor(model, str(image_path))
        prediction = np.asarray(result[0], dtype=np.uint8)
        np.save(args.output_dir / f"{image_path.stem}.npy", prediction)
        if index % 50 == 0:
            print(f"predicted={index}")

    print(f"prediction_dir={args.output_dir}")
    print(f"predictions_written={len(image_paths)}")


if __name__ == "__main__":
    main()
