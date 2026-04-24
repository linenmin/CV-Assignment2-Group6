from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .labels import CLASS_NAMES
from .submission import (
    build_submission_dataframe,
    initialize_placeholder_classification,
    load_test_dataframe,
    rle_encode,
)


@dataclass(frozen=True)
class EnsembleStats:
    num_images: int
    mean_changed_vs_primary: float
    mean_pairwise_disagreement: float


def rle_decode(encoded: str | float, shape: tuple[int, ...]) -> np.ndarray:
    size = int(np.prod(shape))
    mask = np.zeros(size, dtype=np.uint8)
    if not isinstance(encoded, str) or encoded.strip() == "":
        return mask.reshape(shape)

    values = np.asarray(encoded.split(), dtype=np.int64)
    starts = values[0::2] - 1
    lengths = values[1::2]
    for start, length in zip(starts, lengths):
        mask[start : start + length] = 1
    return mask.reshape(shape)


def decode_segmentation_rle(encoded: str | float, image_shape: tuple[int, int]) -> np.ndarray:
    height, width = image_shape
    class_stack = rle_decode(encoded, (len(CLASS_NAMES), height, width))
    prediction = np.zeros((height, width), dtype=np.uint8)
    for class_index in range(len(CLASS_NAMES)):
        prediction[class_stack[class_index].astype(bool)] = class_index + 1
    return prediction


def encode_segmentation_mask(mask: np.ndarray) -> str:
    class_stack = np.array(
        [mask == class_id for class_id in range(1, len(CLASS_NAMES) + 1)],
        dtype=np.uint8,
    )
    return rle_encode(class_stack)


def load_submission(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col="Id", dtype={"Predicted": str}).fillna("")


def vote_masks(masks: list[np.ndarray], tie_break_mask: np.ndarray) -> np.ndarray:
    if not masks:
        raise ValueError("At least one mask is required for voting.")
    stacked = np.stack(masks, axis=0)
    output = tie_break_mask.copy().astype(np.uint8)
    best_counts = np.sum(stacked == output[None, :, :], axis=0)

    for label_id in range(len(CLASS_NAMES) + 1):
        counts = np.sum(stacked == label_id, axis=0)
        improve = counts > best_counts
        output[improve] = label_id
        best_counts[improve] = counts[improve]
    return output


def build_hard_vote_submission(
    submission_paths: list[str | Path],
    output_path: str | Path,
    primary_index: int = -1,
) -> tuple[Path, EnsembleStats]:
    if len(submission_paths) < 2:
        raise ValueError("At least two submissions are required for an ensemble.")

    submissions = [load_submission(path) for path in submission_paths]
    primary = submissions[primary_index]
    test_df = initialize_placeholder_classification(load_test_dataframe(), fill_value=0)

    changed_fractions: list[float] = []
    pairwise_disagreements: list[float] = []
    segmentation_predictions: list[np.ndarray] = []

    for test_id in test_df.index.tolist():
        image = np.load(_test_image_path(test_id), mmap_mode="r")
        height, width = int(image.shape[0]), int(image.shape[1])
        row_id = f"{test_id}_segmentation"

        masks = [
            decode_segmentation_rle(submission.loc[row_id, "Predicted"], (height, width))
            for submission in submissions
        ]
        primary_mask = decode_segmentation_rle(primary.loc[row_id, "Predicted"], (height, width))
        voted = vote_masks(masks, tie_break_mask=primary_mask)
        segmentation_predictions.append(voted)

        changed_fractions.append(float(np.mean(voted != primary_mask)))
        for left_index in range(len(masks)):
            for right_index in range(left_index + 1, len(masks)):
                pairwise_disagreements.append(float(np.mean(masks[left_index] != masks[right_index])))

    test_df["seg"] = segmentation_predictions
    submission_df = build_submission_dataframe(test_df)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_path)

    stats = EnsembleStats(
        num_images=len(test_df),
        mean_changed_vs_primary=float(np.mean(changed_fractions)),
        mean_pairwise_disagreement=float(np.mean(pairwise_disagreements)),
    )
    return output_path, stats


def _test_image_path(test_id: int) -> Path:
    from .submission import get_test_image_dir

    return get_test_image_dir() / f"test_{test_id}.npy"
