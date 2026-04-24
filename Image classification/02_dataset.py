"""
Step 2: PyTorch Dataset for PASCAL VOC 2009 multi-label classification.

Importable by train, evaluate, and predict scripts.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).parent))
from shared import LABELS  # noqa: E402

# ImageNet normalisation constants
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


def get_train_transform(img_size: int = 320) -> T.Compose:
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        T.ToTensor(),
        T.Normalize(_MEAN, _STD),
        # RandomErasing: masks random patches → model learns global rather than
        # local cues; especially helps diningtable/bottle/pottedplant which are
        # often partially occluded.
        T.RandomErasing(p=0.3, scale=(0.02, 0.2)),
    ])


def get_val_transform(img_size: int = 320) -> T.Compose:
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(_MEAN, _STD),
    ])


class VOCDataset(Dataset):
    """
    Parameters
    ----------
    df        : DataFrame with index=Id; must contain LABELS columns for train mode.
    data_dir  : Path to kul-computer-vision-ga-2-2026/
    split     : 'train' or 'test' — determines which img sub-folder to use.
    transform : torchvision transform pipeline; defaults to val_transform.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        data_dir: Path,
        split: str = "train",
        transform=None,
    ):
        self.df = df
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform or get_val_transform()
        self.has_labels = all(c in df.columns for c in LABELS)
        self.indices = list(df.index)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        npy_path = self.data_dir / self.split / "img" / f"{self.split}_{idx}.npy"
        arr = np.load(npy_path)               # (H, W, 3) uint8
        img = Image.fromarray(arr)
        img = self.transform(img)              # (3, H, W) float32

        if self.has_labels:
            label = torch.FloatTensor(self.df.loc[idx, LABELS].values.astype(float))
            return img, label
        return img, idx   # test mode: return original index for alignment


def load_train_df(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(data_dir / "train" / "train_set.csv", index_col="Id")


def load_test_df(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(data_dir / "test" / "test_set.csv", index_col="Id")


# ---------------------------------------------------------------------------
# Quick sanity check when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    data_dir = Path(__file__).parent.parent / "kul-computer-vision-ga-2-2026"
    df = load_train_df(data_dir)
    ds = VOCDataset(df, data_dir, split="train", transform=get_train_transform())
    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)

    imgs, labels = next(iter(loader))
    print(f"Batch image shape : {imgs.shape}")   # (8, 3, 224, 224)
    print(f"Batch label shape : {labels.shape}") # (8, 20)
    print(f"Label dtype       : {labels.dtype}")
    print("Dataset OK.")
