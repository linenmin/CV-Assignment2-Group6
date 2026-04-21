from pathlib import Path

from ga2_seg.paths import get_dataset_root, get_project_root


def test_project_root_resolves_to_semantic_segmentation_dir():
    project_root = get_project_root()
    assert project_root.name == "Semantic segmentation"
    assert project_root.is_dir()


def test_dataset_root_contains_expected_assignment_structure():
    dataset_root = get_dataset_root()
    assert dataset_root == Path(
        r"D:\BaiduNetdiskWorkspace\Leuven\8th\Computer Vision\assignment\Group2\kul-computer-vision-ga-2-2026"
    )
    assert (dataset_root / "train" / "img").is_dir()
    assert (dataset_root / "train" / "seg").is_dir()
    assert (dataset_root / "test" / "img").is_dir()
