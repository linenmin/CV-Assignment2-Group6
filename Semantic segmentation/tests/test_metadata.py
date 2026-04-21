from ga2_seg.labels import CLASS_NAMES, NUM_CLASSES_WITH_BACKGROUND
from ga2_seg.split import get_split_dir, get_train_split_path, get_val_split_path


def test_class_metadata_matches_assignment_labels():
    assert len(CLASS_NAMES) == 20
    assert CLASS_NAMES[0] == "aeroplane"
    assert CLASS_NAMES[-1] == "tvmonitor"
    assert NUM_CLASSES_WITH_BACKGROUND == 21


def test_split_helpers_resolve_inside_project_data_dir():
    split_dir = get_split_dir()
    assert split_dir.parts[-2:] == ("data", "splits")
    assert get_train_split_path().name == "train.txt"
    assert get_val_split_path().name == "val.txt"
