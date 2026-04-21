"""
Step 1: Data Exploration + Convert all .npy images to PNG

Run from project root:
    python "Image classification/01_explore_data.py"

Outputs:
    output/train_png/train_{idx}.png  - all 750 training images
    output/test_png/test_{idx}.png    - all 750 test images
    output/sample_grid.png            - 20-class × 3-sample overview grid
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from shared import DATA_DIR, OUTPUT_DIR, LABELS  # noqa: E402

TRAIN_PNG_DIR = OUTPUT_DIR / "train_png"
TEST_PNG_DIR  = OUTPUT_DIR / "test_png"


def main():
    for d in (TRAIN_PNG_DIR, TEST_PNG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(DATA_DIR / "train" / "train_set.csv", index_col="Id")
    test_df  = pd.read_csv(DATA_DIR / "test"  / "test_set.csv",  index_col="Id")
    print(f"Training samples : {len(train_df)}")
    print(f"Test samples     : {len(test_df)}")
    print(f"Classes ({len(LABELS)}): {LABELS}\n")

    # 1. Convert ALL training .npy → PNG
    print("=== Converting training images (.npy → PNG) ===")
    train_shapes = []
    for idx in tqdm(train_df.index, desc="train"):
        arr = np.load(DATA_DIR / "train" / "img" / f"train_{idx}.npy")
        train_shapes.append(arr.shape)
        Image.fromarray(arr).save(TRAIN_PNG_DIR / f"train_{idx}.png")
    print(f"Saved {len(train_df)} PNG files to {TRAIN_PNG_DIR}\n")

    # 2. Convert ALL test .npy → PNG
    print("=== Converting test images (.npy → PNG) ===")
    for idx in tqdm(test_df.index, desc="test"):
        arr = np.load(DATA_DIR / "test" / "img" / f"test_{idx}.npy")
        Image.fromarray(arr).save(TEST_PNG_DIR / f"test_{idx}.png")
    print(f"Saved {len(test_df)} PNG files to {TEST_PNG_DIR}\n")

    # 3. Statistical analysis
    heights = [s[0] for s in train_shapes]
    widths  = [s[1] for s in train_shapes]
    print("=== Image Size Statistics (training set) ===")
    print(f"Height — min:{min(heights)}  max:{max(heights)}  mean:{np.mean(heights):.1f}")
    print(f"Width  — min:{min(widths)}   max:{max(widths)}   mean:{np.mean(widths):.1f}\n")

    print("=== Label Distribution (training set) ===")
    label_counts = train_df[LABELS].sum().sort_values(ascending=False)
    for label, count in label_counts.items():
        bar = "█" * int(count / len(train_df) * 40)
        print(f"  {label:<15s} {count:4d} ({count/len(train_df)*100:5.1f}%)  {bar}")

    labels_per_image = train_df[LABELS].sum(axis=1)
    print(f"\nLabels per image — min:{labels_per_image.min()}  max:{labels_per_image.max()}  "
          f"mean:{labels_per_image.mean():.2f}")
    print(f"Single-label images : {(labels_per_image == 1).sum()}")
    print(f"Multi-label images  : {(labels_per_image > 1).sum()}\n")

    # 4. Sample overview grid: 20 classes × 3 examples
    SAMPLES_PER_CLASS = 3
    fig, axes = plt.subplots(len(LABELS), SAMPLES_PER_CLASS,
                             figsize=(SAMPLES_PER_CLASS * 3, len(LABELS) * 2.5))
    fig.suptitle("PASCAL VOC 2009 — Training samples per class", fontsize=14, y=1.01)

    for row, label in enumerate(LABELS):
        positive_idx = train_df.index[train_df[label] == 1].tolist()
        samples = positive_idx[:SAMPLES_PER_CLASS]
        for col in range(SAMPLES_PER_CLASS):
            ax = axes[row][col]
            if col < len(samples):
                img = np.array(Image.open(TRAIN_PNG_DIR / f"train_{samples[col]}.png"))
                ax.imshow(img)
            else:
                ax.axis("off")
            if col == 0:
                ax.set_ylabel(label, fontsize=9, rotation=0, labelpad=60, va="center")
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    grid_path = OUTPUT_DIR / "sample_grid.png"
    plt.savefig(grid_path, bbox_inches="tight", dpi=80)
    plt.close()
    print(f"Sample grid saved to {grid_path}")
    print("Done.")


if __name__ == "__main__":
    main()
