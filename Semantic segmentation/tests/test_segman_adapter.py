from ga2_seg.labels import ALL_CLASSES
from ga2_seg.segman_adapter import build_segman_config_text


def test_build_segman_config_text_uses_ga2_classes_and_no_zero_reduction(tmp_path):
    checkpoint = tmp_path / "SegMAN_Encoder_b.pth.tar"
    export_root = tmp_path / "segman_ga2"

    config_text = build_segman_config_text(
        variant="b",
        export_root=export_root,
        encoder_checkpoint=checkpoint,
        max_iters=12,
        val_interval=3,
        batch_size=1,
        workers=0,
    )

    assert "type='SegMANEncoder_b'" in config_text
    assert f"num_classes={len(ALL_CLASSES)}" in config_text
    assert "reduce_zero_label=False" in config_text
    assert "runner = dict(type='IterBasedRunner', max_iters=12)" in config_text
    assert "evaluation = dict(interval=3, metric='mIoU', save_best='mIoU')" in config_text


def test_build_segman_config_text_supports_large_variant(tmp_path):
    config_text = build_segman_config_text(
        variant="l",
        export_root=tmp_path / "segman_ga2",
        encoder_checkpoint=tmp_path / "SegMAN_Encoder_l.pth.tar",
    )

    assert "type='SegMANEncoder_l'" in config_text
    assert "in_channels=[96, 192, 432, 640]" in config_text
