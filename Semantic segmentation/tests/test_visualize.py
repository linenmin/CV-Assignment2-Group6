import numpy as np

from ga2_seg.paths import get_project_root
from ga2_seg.visualize import create_overlay, generate_palette


def test_generate_palette_covers_background_and_all_classes():
    palette = generate_palette(21)
    assert palette.shape == (21, 3)
    assert palette.dtype == np.uint8


def test_create_overlay_preserves_image_shape():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 3
    overlay = create_overlay(image, mask)
    assert overlay.shape == image.shape
    assert overlay.dtype == np.uint8


def test_visualize_samples_script_writes_expected_pngs(tmp_path):
    from scripts.visualize_samples import generate_visualization_outputs

    output_dir = tmp_path / "figures"
    generated = generate_visualization_outputs(output_dir=output_dir, num_samples=2)
    expected = {"sample_grid.png", "transform_preview.png"}
    assert expected.issubset({path.name for path in generated})
