"""Side-by-side comparison of the 'train' class in VOC GA2 vs Cityscapes.

Picks the strongest 'train' examples from each dataset (largest train-mask
area) and writes single PNGs combining: image + GT mask overlay where
train pixels are highlighted in red.

Used to visually explain why V17 transfers near-zero IoU for the train
class to Cityscapes despite scoring well on VOC train images.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


VOC_TRAIN_LABEL_ID = 19
CITYSCAPES_TRAIN_LABEL_ID = 31

VOC_DATA_ROOT = Path("../kul-computer-vision-ga-2-2026")
CITYSCAPES_LEFT_IMG_ROOT = Path("G:/Datasets/leftImg8bit_trainvaltest/leftImg8bit/val")
CITYSCAPES_GT_ROOT = Path("G:/Datasets/gtFine_trainvaltest/gtFine/val")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/figures/train_class_voc_vs_cityscapes"),
    )
    parser.add_argument("--num-per-dataset", type=int, default=4)
    parser.add_argument("--panel-width", type=int, default=640)
    return parser.parse_args()


def overlay_mask(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    base = image.convert("RGB")
    overlay = Image.new("RGB", base.size, color)
    alpha_arr = (mask > 0).astype(np.uint8) * 140
    alpha = Image.fromarray(alpha_arr, mode="L").resize(base.size, Image.Resampling.NEAREST)
    return Image.composite(overlay, base, alpha)


def annotate(image: Image.Image, text: str) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    pad = 6
    bbox = draw.textbbox((pad, pad), text, font=font)
    draw.rectangle(bbox, fill=(0, 0, 0))
    draw.text((pad, pad), text, fill=(255, 255, 255), font=font)
    return out


def resize_keep_width(image: Image.Image, width: int) -> Image.Image:
    w, h = image.size
    if w == width:
        return image
    new_h = int(round(h * width / w))
    return image.resize((width, new_h), Image.Resampling.LANCZOS)


def voc_candidates() -> list[tuple[int, int]]:
    """Return (train_pixels, sample_id) sorted descending."""
    import pandas as pd

    df = pd.read_csv(VOC_DATA_ROOT / "train" / "train_set.csv", index_col="Id")
    ids = df.index[df["train"] == 1].tolist()
    candidates: list[tuple[int, int]] = []
    for sid in ids:
        mask = np.load(VOC_DATA_ROOT / "train" / "seg" / f"train_{sid}.npy")
        candidates.append((int((mask == VOC_TRAIN_LABEL_ID).sum()), sid))
    candidates.sort(reverse=True)
    return candidates


def cityscapes_candidates() -> list[tuple[int, Path, Path]]:
    """Return (train_pixels, image_path, label_path) sorted descending."""
    candidates: list[tuple[int, Path, Path]] = []
    for label_path in sorted(CITYSCAPES_GT_ROOT.glob("*/*_gtFine_labelIds.png")):
        arr = np.asarray(Image.open(label_path))
        count = int((arr == CITYSCAPES_TRAIN_LABEL_ID).sum())
        if count == 0:
            continue
        city = label_path.parent.name
        stem = label_path.name.replace("_gtFine_labelIds.png", "")
        image_path = CITYSCAPES_LEFT_IMG_ROOT / city / f"{stem}_leftImg8bit.png"
        if image_path.exists():
            candidates.append((count, image_path, label_path))
    candidates.sort(reverse=True, key=lambda triple: triple[0])
    return candidates


def make_voc_panel(sample_id: int, train_pixels: int, panel_width: int) -> Image.Image:
    image_arr = np.load(VOC_DATA_ROOT / "train" / "img" / f"train_{sample_id}.npy")
    image = Image.fromarray(image_arr.astype(np.uint8))
    mask = np.load(VOC_DATA_ROOT / "train" / "seg" / f"train_{sample_id}.npy")
    train_mask = (mask == VOC_TRAIN_LABEL_ID).astype(np.uint8)
    overlay = overlay_mask(image, train_mask, (255, 0, 0))
    image = resize_keep_width(image, panel_width)
    overlay = resize_keep_width(overlay, panel_width)
    image = annotate(image, f"VOC GA2  train_{sample_id}.png  ({image.size[0]}x{image.size[1]})")
    overlay = annotate(overlay, f"GT train pixels: {train_pixels}")
    canvas = Image.new("RGB", (image.size[0], image.size[1] + overlay.size[1]))
    canvas.paste(image, (0, 0))
    canvas.paste(overlay, (0, image.size[1]))
    return canvas


def make_cityscapes_panel(
    image_path: Path, label_path: Path, train_pixels: int, panel_width: int
) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    label = np.asarray(Image.open(label_path))
    train_mask = (label == CITYSCAPES_TRAIN_LABEL_ID).astype(np.uint8)
    overlay = overlay_mask(image, train_mask, (255, 0, 0))
    image = resize_keep_width(image, panel_width)
    overlay = resize_keep_width(overlay, panel_width)
    image = annotate(image, f"Cityscapes  {image_path.stem} ({image.size[0]}x{image.size[1]})")
    overlay = annotate(overlay, f"GT train pixels: {train_pixels}")
    canvas = Image.new("RGB", (image.size[0], image.size[1] + overlay.size[1]))
    canvas.paste(image, (0, 0))
    canvas.paste(overlay, (0, image.size[1]))
    return canvas


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    voc = voc_candidates()[: args.num_per_dataset]
    cs = cityscapes_candidates()[: args.num_per_dataset]
    n = min(len(voc), len(cs))
    if n == 0:
        raise RuntimeError("No candidates found in at least one dataset.")

    panel_width = args.panel_width
    summary_lines = [
        "# 'train' class visual comparison: VOC GA2 vs Cityscapes",
        "",
        "Row = matched pair (left VOC, right Cityscapes), ordered by Cityscapes train area.",
        "Each panel is image-on-top, GT train-mask overlay below.",
        "",
    ]

    for rank in range(n):
        voc_pixels, voc_sid = voc[rank]
        cs_pixels, cs_img, cs_lbl = cs[rank]

        voc_panel = make_voc_panel(voc_sid, voc_pixels, panel_width)
        cs_panel = make_cityscapes_panel(cs_img, cs_lbl, cs_pixels, panel_width)

        height = max(voc_panel.size[1], cs_panel.size[1])
        canvas = Image.new("RGB", (panel_width * 2 + 12, height), (30, 30, 30))
        canvas.paste(voc_panel, (0, 0))
        canvas.paste(cs_panel, (panel_width + 12, 0))

        out_path = args.output_dir / f"pair_{rank:02d}_voc{voc_sid}_vs_{cs_img.stem}.png"
        canvas.save(out_path)
        summary_lines.append(
            f"- `{out_path.name}`: VOC train_{voc_sid} ({voc_pixels} px) vs "
            f"Cityscapes {cs_img.stem} ({cs_pixels} px)"
        )
        print(f"wrote {out_path}", flush=True)

    (args.output_dir / "README.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"output_dir={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
