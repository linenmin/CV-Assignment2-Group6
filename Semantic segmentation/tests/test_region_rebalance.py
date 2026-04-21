import torch

from ga2_seg.region_rebalance import build_region_class_frequencies
from ga2_seg.region_rebalance_head import RegionRebalanceLightHamHead


def test_region_class_frequencies_match_train_split_length():
    frequencies = build_region_class_frequencies()
    assert len(frequencies) == 20
    assert frequencies[14] == 168
    assert min(frequencies) > 0


def test_region_rebalance_head_emits_auxiliary_loss():
    head = RegionRebalanceLightHamHead(
        in_channels=[4, 8, 16],
        in_index=[0, 1, 2],
        channels=8,
        ham_channels=8,
        num_classes=3,
        norm_cfg=None,
        align_corners=False,
        dropout_ratio=0.0,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
        region_class_frequencies=[3, 2],
        region_loss_weight=0.5,
        include_background_in_region_branch=False,
        region_min_pixels=1,
    )

    class _SegData:
        def __init__(self, data):
            self.data = data

    class _Sample:
        def __init__(self, label):
            self.gt_sem_seg = _SegData(label)

    inputs = (
        torch.randn(1, 4, 32, 32),
        torch.randn(1, 8, 16, 16),
        torch.randn(1, 16, 8, 8),
    )
    label = torch.zeros(1, 32, 32, dtype=torch.long)
    label[:, 8:16, 8:16] = 1
    label[:, 16:24, 16:24] = 2

    losses = head.loss(inputs, [_Sample(label)], train_cfg={})
    assert "loss_ce" in losses
    assert "loss_region_rebalance" in losses
    assert float(losses["loss_region_rebalance"].detach()) > 0.0
