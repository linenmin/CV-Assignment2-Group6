from pathlib import Path

import numpy as np
import pandas as pd

from .labels import CLASS_NAMES
from .paths import get_dataset_root, get_project_root


def get_test_csv_path() -> Path:
    return get_dataset_root() / "test" / "test_set.csv"


def get_test_image_dir() -> Path:
    return get_dataset_root() / "test" / "img"


def get_submission_output_dir() -> Path:
    return get_project_root() / "outputs" / "submissions"


def load_test_dataframe(limit: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(get_test_csv_path(), index_col="Id")
    if limit is not None:
        df = df.iloc[:limit].copy()
    return df


def iter_test_images(limit: int | None = None):
    test_df = load_test_dataframe(limit=limit)
    image_dir = get_test_image_dir()
    for test_id in test_df.index.tolist():
        yield int(test_id), np.load(image_dir / f"test_{test_id}.npy")


def initialize_placeholder_classification(
    test_df: pd.DataFrame,
    fill_value: int = 0,
) -> pd.DataFrame:
    filled_df = test_df.copy()
    for label in CLASS_NAMES:
        filled_df[label] = fill_value
    return filled_df


def attach_segmentation_predictions(
    test_df: pd.DataFrame,
    prediction_dir: str | Path,
) -> pd.DataFrame:
    prediction_dir = Path(prediction_dir)
    attached_df = test_df.copy()
    attached_df["seg"] = [
        np.load(prediction_dir / f"test_{test_id}.npy") for test_id in attached_df.index.tolist()
    ]
    return attached_df


def rle_encode(img: np.ndarray) -> str:
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def build_submission_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df_dict = {"Id": [], "Predicted": []}
    for idx in df.index.tolist():
        class_prediction = np.asarray(df.loc[idx, list(CLASS_NAMES)], dtype=np.uint8)
        seg_prediction = np.asarray(df.loc[idx, "seg"], dtype=np.uint8)

        df_dict["Id"].append(f"{idx}_classification")
        df_dict["Predicted"].append(rle_encode(class_prediction))
        df_dict["Id"].append(f"{idx}_segmentation")
        df_dict["Predicted"].append(
            rle_encode(np.array([seg_prediction == class_id for class_id in range(1, len(CLASS_NAMES) + 1)]))
        )

    return pd.DataFrame(data=df_dict, dtype=str).set_index("Id")


def write_submission_csv(
    submission_df: pd.DataFrame,
    output_path: str | Path | None = None,
) -> Path:
    output_path = (
        get_submission_output_dir() / "submission.csv" if output_path is None else Path(output_path)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_path)
    return output_path
