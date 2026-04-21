import json

from ga2_seg.labels import CLASS_NAMES
from ga2_seg.paths import get_project_root
from ga2_seg.stats import (
    compute_class_presence_counts,
    compute_image_shape_counts,
    compute_pixel_label_counts,
    load_train_labels_df,
)


def test_load_train_labels_df_matches_assignment_labels():
    train_df = load_train_labels_df()
    assert list(train_df.columns) == list(CLASS_NAMES)
    assert len(train_df) == 749


def test_image_shape_counts_are_non_empty_and_include_common_shape():
    shape_counts = compute_image_shape_counts()
    assert shape_counts
    assert "(375, 500, 3)" in shape_counts


def test_pixel_label_counts_include_background_and_foreground():
    pixel_counts = compute_pixel_label_counts()
    assert "0" in pixel_counts
    assert "15" in pixel_counts
    assert pixel_counts["0"] > pixel_counts["15"] > 0


def test_class_presence_counts_include_person_and_sheep():
    class_counts = compute_class_presence_counts(load_train_labels_df())
    assert class_counts["person"] > class_counts["sheep"] > 0


def test_analysis_script_outputs_stats_json(tmp_path):
    from scripts.analyze_dataset import generate_analysis_outputs

    output_dir = tmp_path / "figures"
    metadata_dir = tmp_path / "metadata"
    generate_analysis_outputs(output_dir=output_dir, metadata_dir=metadata_dir)

    stats_path = metadata_dir / "stats.json"
    assert stats_path.is_file()
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    assert stats["train_examples"] == 749
    assert stats["num_classes"] == 20
