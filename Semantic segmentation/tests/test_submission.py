import numpy as np
import pandas as pd

from ga2_seg.labels import CLASS_NAMES
from ga2_seg.submission import (
    attach_segmentation_predictions,
    build_submission_dataframe,
    rle_encode,
)


def test_rle_encode_matches_template_behavior():
    binary = np.array([0, 1, 1, 0, 1], dtype=np.uint8)
    assert rle_encode(binary) == "2 2 5 1"


def test_build_submission_dataframe_creates_classification_and_segmentation_rows():
    df = pd.DataFrame([{label: 0 for label in CLASS_NAMES}], index=[7])
    df.loc[7, "aeroplane"] = 1
    df.loc[7, "bus"] = 1
    df["seg"] = [np.array([[0, 1], [2, 1]], dtype=np.uint8)]

    submission_df = build_submission_dataframe(df)

    assert submission_df.index.tolist() == ["7_classification", "7_segmentation"]
    assert submission_df.loc["7_classification", "Predicted"] == "1 1 6 1"
    assert submission_df.loc["7_segmentation", "Predicted"] == "2 1 4 1 7 1"


def test_attach_segmentation_predictions_loads_saved_masks(tmp_path):
    df = pd.DataFrame([{label: 0 for label in CLASS_NAMES}], index=[3])
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()

    expected_mask = np.array([[1, 0], [0, 2]], dtype=np.uint8)
    np.save(prediction_dir / "test_3.npy", expected_mask)

    attached_df = attach_segmentation_predictions(df, prediction_dir)

    assert "seg" in attached_df.columns
    np.testing.assert_array_equal(attached_df.loc[3, "seg"], expected_mask)
