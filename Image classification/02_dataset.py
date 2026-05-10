"""Step 2: PyTorch dataset and transforms for VOC multi-label classification."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageOps
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).parent))
from shared import LABELS  # noqa: E402

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]
_PAD_FILL = tuple(int(round(channel * 255)) for channel in _MEAN)


class SquarePad:
    """Pad a PIL image to a square while preserving aspect ratio."""

    def __init__(self, fill: tuple[int, int, int] = _PAD_FILL):
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        width, height = img.size
        side = max(width, height)
        pad_left = (side - width) // 2
        pad_top = (side - height) // 2
        pad_right = side - width - pad_left
        pad_bottom = side - height - pad_top
        return ImageOps.expand(
            img,
            border=(pad_left, pad_top, pad_right, pad_bottom),
            fill=self.fill,
        )


def _resize_ops(img_size: int, mode: str) -> list:
    if mode == "resize":
        return [T.Resize((img_size, img_size))]
    if mode == "square_pad":
        return [SquarePad(), T.Resize((img_size, img_size))]
    raise ValueError(f"Unknown transform mode: {mode}")


def get_train_transform(img_size: int = 320, mode: str = "resize") -> T.Compose:
    return T.Compose(
        _resize_ops(img_size, mode)
        + [
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            T.ToTensor(),
            T.Normalize(_MEAN, _STD),
            T.RandomErasing(p=0.3, scale=(0.02, 0.2)),
        ]
    )


def get_val_transform(img_size: int = 320, mode: str = "resize") -> T.Compose:
    return T.Compose(
        _resize_ops(img_size, mode)
        + [
            T.ToTensor(),
            T.Normalize(_MEAN, _STD),
        ]
    )


class VOCDataset(Dataset):
    """Dataset backed by the assignment's .npy image files."""

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
        self.has_labels = all(column in df.columns for column in LABELS)
        self.indices = list(df.index)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        npy_path = self.data_dir / self.split / "img" / f"{self.split}_{idx}.npy"
        arr = np.load(npy_path)
        img = self.transform(Image.fromarray(arr))

        if self.has_labels:
            label = torch.FloatTensor(self.df.loc[idx, LABELS].values.astype(float))
            return img, label
        return img, idx


def load_train_df(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(data_dir / "train" / "train_set.csv", index_col="Id")


def load_test_df(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(data_dir / "test" / "test_set.csv", index_col="Id")


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    data_dir = Path(__file__).parent.parent / "kul-computer-vision-ga-2-2026"
    df = load_train_df(data_dir)
    ds = VOCDataset(df, data_dir, split="train", transform=get_train_transform())
    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)

    imgs, labels = next(iter(loader))
    print(f"Batch image shape : {imgs.shape}")
    print(f"Batch label shape : {labels.shape}")
    print(f"Label dtype       : {labels.dtype}")
    print("Dataset OK.")
