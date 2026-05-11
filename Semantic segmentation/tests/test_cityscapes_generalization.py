from pathlib import Path

import numpy as np

from ga2_seg.cityscapes_generalization import (
    IGNORE_LABEL,
    collect_cityscapes_pairs,
    compute_overlap_intersections_unions,
    map_cityscapes_label_ids,
    resolve_cityscapes_roots,
)
from ga2_seg.labels import CLASS_NAME_TO_LABEL_ID


def test_map_cityscapes_label_ids_maps_overlap_classes() -> None:
    labels = np.array([[24, 26, 28], [31, 32, 33], [7, 8, 11]], dtype=np.uint8)

    mapped = map_cityscapes_label_ids(labels)

    assert mapped[0, 0] == CLASS_NAME_TO_LABEL_ID["person"]
    assert mapped[0, 1] == CLASS_NAME_TO_LABEL_ID["car"]
    assert mapped[0, 2] == CLASS_NAME_TO_LABEL_ID["bus"]
    assert mapped[1, 0] == CLASS_NAME_TO_LABEL_ID["train"]
    assert mapped[1, 1] == CLASS_NAME_TO_LABEL_ID["motorbike"]
    assert mapped[1, 2] == CLASS_NAME_TO_LABEL_ID["bicycle"]
    assert np.all(mapped[2] == IGNORE_LABEL)


def test_collect_cityscapes_pairs_accepts_trainvaltest_outer_dirs(tmp_path: Path) -> None:
    left_city = tmp_path / "leftImg8bit_trainvaltest" / "leftImg8bit" / "val" / "frankfurt"
    gt_city = tmp_path / "gtFine_trainvaltest" / "gtFine" / "val" / "frankfurt"
    left_city.mkdir(parents=True)
    gt_city.mkdir(parents=True)
    (left_city / "frankfurt_000000_000294_leftImg8bit.png").write_bytes(b"fake")
    (gt_city / "frankfurt_000000_000294_gtFine_labelIds.png").write_bytes(b"fake")

    roots = resolve_cityscapes_roots(
        left_img_root=tmp_path / "leftImg8bit_trainvaltest",
        gt_fine_root=tmp_path / "gtFine_trainvaltest",
    )
    pairs = collect_cityscapes_pairs(roots)

    assert len(pairs) == 1
    assert pairs[0].sample_id == "frankfurt_000000_000294"


def test_compute_overlap_intersections_unions_ignores_non_overlap_pixels() -> None:
    car_id = CLASS_NAME_TO_LABEL_ID["car"]
    bus_id = CLASS_NAME_TO_LABEL_ID["bus"]
    prediction = np.array([[car_id, car_id], [bus_id, bus_id]], dtype=np.uint8)
    target = np.array([[car_id, IGNORE_LABEL], [bus_id, car_id]], dtype=np.uint8)

    intersections, unions, target_pixels = compute_overlap_intersections_unions(
        prediction=prediction,
        target=target,
        class_ids=(car_id, bus_id),
    )

    assert intersections[car_id] == 1
    assert unions[car_id] == 2
    assert target_pixels[car_id] == 2
    assert intersections[bus_id] == 1
    assert unions[bus_id] == 2
    assert target_pixels[bus_id] == 1
