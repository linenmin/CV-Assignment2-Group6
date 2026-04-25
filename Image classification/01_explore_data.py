"""Step 1: inspect the classification data and save a sample grid.

Run from the project root:
    python "Image classification/01_explore_data.py"
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from shared import DATA_DIR, DATA_EXPORT_DIR, LABELS  # noqa: E402

FIGURE_DIR = DATA_EXPORT_DIR / "figures"


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(DATA_DIR / "train" / "train_set.csv", index_col="Id")
    test_df = pd.read_csv(DATA_DIR / "test" / "test_set.csv", index_col="Id")
    print(f"Training samples : {len(train_df)}")
    print(f"Test samples     : {len(test_df)}")
    print(f"Classes ({len(LABELS)}): {LABELS}\n")

    print("=== Reading training image shapes ===")
    train_shapes = []
    for idx in tqdm(train_df.index, desc="train"):
        arr = np.load(DATA_DIR / "train" / "img" / f"train_{idx}.npy")
        train_shapes.append(arr.shape)

    heights = [shape[0] for shape in train_shapes]
    widths = [shape[1] for shape in train_shapes]
    print("=== Image Size Statistics (training set) ===")
    print(f"Height min:{min(heights)}  max:{max(heights)}  mean:{np.mean(heights):.1f}")
    print(f"Width  min:{min(widths)}   max:{max(widths)}   mean:{np.mean(widths):.1f}\n")

    print("=== Label Distribution (training set) ===")
    label_counts = train_df[LABELS].sum().sort_values(ascending=False)
    for label, count in label_counts.items():
        bar = "#" * int(count / len(train_df) * 40)
        print(f"  {label:<15s} {count:4d} ({count / len(train_df) * 100:5.1f}%)  {bar}")

    labels_per_image = train_df[LABELS].sum(axis=1)
    print(
        f"\nLabels per image min:{labels_per_image.min()}  "
        f"max:{labels_per_image.max()}  mean:{labels_per_image.mean():.2f}"
    )
    print(f"Single-label images : {(labels_per_image == 1).sum()}")
    print(f"Multi-label images  : {(labels_per_image > 1).sum()}\n")

    samples_per_class = 3
    fig, axes = plt.subplots(
        len(LABELS),
        samples_per_class,
        figsize=(samples_per_class * 3, len(LABELS) * 2.5),
    )
    fig.suptitle("PASCAL VOC 2009 training samples per class", fontsize=14, y=1.01)

    for row, label in enumerate(LABELS):
        positive_idx = train_df.index[train_df[label] == 1].tolist()
        samples = positive_idx[:samples_per_class]
        for col in range(samples_per_class):
            ax = axes[row][col]
            if col < len(samples):
                img = np.load(DATA_DIR / "train" / "img" / f"train_{samples[col]}.npy")
                ax.imshow(img)
            else:
                ax.axis("off")
            if col == 0:
                ax.set_ylabel(label, fontsize=9, rotation=0, labelpad=60, va="center")
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    grid_path = FIGURE_DIR / "sample_grid.png"
    plt.savefig(grid_path, bbox_inches="tight", dpi=80)
    plt.close()
    print(f"Sample grid saved to {grid_path}")
    print("Done.")


if __name__ == "__main__":
    main()
