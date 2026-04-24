import numpy as np

from ga2_seg.submission_ensemble import (
    decode_segmentation_rle,
    encode_segmentation_mask,
    rle_decode,
    vote_masks,
)


def test_rle_decode_handles_empty_prediction():
    decoded = rle_decode("", (2, 3))
    assert decoded.shape == (2, 3)
    assert decoded.sum() == 0


def test_segmentation_encode_decode_round_trip():
    mask = np.array(
        [
            [0, 1, 2],
            [3, 0, 1],
        ],
        dtype=np.uint8,
    )
    encoded = encode_segmentation_mask(mask)
    decoded = decode_segmentation_rle(encoded, mask.shape)
    np.testing.assert_array_equal(decoded, mask)


def test_vote_masks_uses_primary_for_ties():
    primary = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    other_a = np.array([[1, 5], [6, 4]], dtype=np.uint8)
    other_b = np.array([[2, 7], [6, 8]], dtype=np.uint8)

    voted = vote_masks([primary, other_a, other_b], tie_break_mask=primary)

    expected = np.array([[1, 2], [6, 4]], dtype=np.uint8)
    np.testing.assert_array_equal(voted, expected)
