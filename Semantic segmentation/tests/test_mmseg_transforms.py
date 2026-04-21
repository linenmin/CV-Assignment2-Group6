import numpy as np

from ga2_seg.mmseg_transforms import RareClassFocusedCrop


def test_rare_class_focused_crop_centers_on_target_class_pixels():
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    seg_map = np.zeros((640, 640), dtype=np.uint8)
    seg_map[520:560, 520:560] = 17

    transform = RareClassFocusedCrop(
        crop_size=(256, 256),
        rare_class_ids=[17],
        rare_prob=1.0,
        cat_max_ratio=1.0,
        max_attempts=5,
    )
    results = transform.transform(
        {
            "img": img,
            "gt_seg_map": seg_map,
            "seg_fields": ["gt_seg_map"],
        }
    )

    assert results["img"].shape[:2] == (256, 256)
    assert np.any(results["gt_seg_map"] == 17)


def test_rare_class_focused_crop_falls_back_to_random_when_target_absent():
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    seg_map = np.zeros((640, 640), dtype=np.uint8)

    transform = RareClassFocusedCrop(
        crop_size=(256, 256),
        rare_class_ids=[17],
        rare_prob=1.0,
        cat_max_ratio=1.0,
        max_attempts=5,
    )
    results = transform.transform(
        {
            "img": img,
            "gt_seg_map": seg_map,
            "seg_fields": ["gt_seg_map"],
        }
    )

    assert results["img"].shape[:2] == (256, 256)
    assert results["gt_seg_map"].shape == (256, 256)
