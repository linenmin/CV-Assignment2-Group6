from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .paths import get_dataset_root


def generate_palette(num_classes: int) -> np.ndarray:
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for class_id in range(num_classes):
        palette[class_id] = (
            (class_id * 37) % 255,
            (class_id * 67) % 255,
            (class_id * 97) % 255,
        )
    return palette


def colorize_mask(mask: np.ndarray, palette: np.ndarray | None = None) -> np.ndarray:
    palette = generate_palette(int(mask.max()) + 1) if palette is None else palette
    return palette[mask]


def create_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    palette = generate_palette(int(mask.max()) + 1)
    colored_mask = colorize_mask(mask, palette)
    overlay = image.astype(np.float32) * (1 - alpha) + colored_mask.astype(np.float32) * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def resize_pair(image: np.ndarray, mask: np.ndarray, size: tuple[int, int] = (512, 512)) -> tuple[np.ndarray, np.ndarray]:
    image_resized = np.array(Image.fromarray(image).resize(size, resample=Image.BILINEAR))
    mask_resized = np.array(Image.fromarray(mask).resize(size, resample=Image.NEAREST))
    return image_resized, mask_resized


def center_crop_pair(image: np.ndarray, mask: np.ndarray, crop_size: tuple[int, int] = (384, 384)) -> tuple[np.ndarray, np.ndarray]:
    crop_h, crop_w = crop_size
    height, width = image.shape[:2]
    top = max((height - crop_h) // 2, 0)
    left = max((width - crop_w) // 2, 0)
    return image[top : top + crop_h, left : left + crop_w], mask[top : top + crop_h, left : left + crop_w]


def photometric_distortion(image: np.ndarray) -> np.ndarray:
    adjusted = image.astype(np.float32) * 1.1 + 12.0
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def load_train_sample(sample_index: int) -> tuple[np.ndarray, np.ndarray]:
    dataset_root = get_dataset_root()
    image = np.load(dataset_root / "train" / "img" / f"train_{sample_index}.npy")
    mask = np.load(dataset_root / "train" / "seg" / f"train_{sample_index}.npy")
    return image, mask


def save_sample_grid(output_path: Path, sample_indices: list[int]) -> Path:
    fig, axes = plt.subplots(len(sample_indices), 3, figsize=(12, 4 * len(sample_indices)))
    if len(sample_indices) == 1:
        axes = np.expand_dims(axes, axis=0)
    for row, sample_index in enumerate(sample_indices):
        image, mask = load_train_sample(sample_index)
        overlay = create_overlay(image, mask)
        axes[row, 0].imshow(image)
        axes[row, 0].set_title(f"Image {sample_index}")
        axes[row, 1].imshow(colorize_mask(mask))
        axes[row, 1].set_title("Mask")
        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title("Overlay")
        for col in range(3):
            axes[row, col].axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def save_transform_preview(output_path: Path, sample_index: int = 0) -> Path:
    image, mask = load_train_sample(sample_index)
    resized_image, resized_mask = resize_pair(image, mask)
    cropped_image, cropped_mask = center_crop_pair(resized_image, resized_mask)
    flipped_image = np.ascontiguousarray(cropped_image[:, ::-1])
    flipped_mask = np.ascontiguousarray(cropped_mask[:, ::-1])
    distorted_image = photometric_distortion(flipped_image)

    panels = [
        ("Raw", image),
        ("Raw Overlay", create_overlay(image, mask)),
        ("Resized Overlay", create_overlay(resized_image, resized_mask)),
        ("Cropped Overlay", create_overlay(cropped_image, cropped_mask)),
        ("Flipped Overlay", create_overlay(flipped_image, flipped_mask)),
        ("Distorted Overlay", create_overlay(distorted_image, flipped_mask)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for ax, (title, panel) in zip(axes.flat, panels):
        ax.imshow(panel)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path

