"""
Step 3: Multi-label classifier with swappable backbone.

Supports:
  backbone='efficientnet_b3'  — 1536-dim features, 81.6% ImageNet top-1  (default)
  backbone='resnet50'         — 2048-dim features, 76.1% ImageNet top-1

Both expose the same API: freeze_backbone / unfreeze_backbone / forward.
"""

import torch
import torch.nn as nn
import torchvision.models as models

_BACKBONE_CONFIGS = {
    "resnet50": {
        "factory":    lambda pretrained: models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        ),
        "feat_dim":   2048,
        # strip the final FC; everything up to AdaptiveAvgPool is kept
        "features_fn": lambda m: nn.Sequential(*list(m.children())[:-1]),
    },
    "efficientnet_b3": {
        "factory":    lambda pretrained: models.efficientnet_b3(
            weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        ),
        "feat_dim":   1536,
        # keep the pretrained features + avgpool, drop the classifier
        "features_fn": lambda m: nn.Sequential(m.features, m.avgpool),
    },
}


class MultiLabelClassifier(nn.Module):
    """
    Parameters
    ----------
    backbone    : 'efficientnet_b3' (default) or 'resnet50'
    num_classes : number of output labels (20 for PASCAL VOC)
    pretrained  : load ImageNet weights for the backbone
    """

    def __init__(self, backbone: str = "efficientnet_b3",
                 num_classes: int = 20, pretrained: bool = True):
        super().__init__()
        cfg = _BACKBONE_CONFIGS[backbone]
        m   = cfg["factory"](pretrained)
        self.features   = cfg["features_fn"](m)
        self._feat_dim  = cfg["feat_dim"]

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(self._feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes),
        )
        for layer in self.classifier.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def freeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad = True

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)   # (B, feat_dim, 1, 1)
        x = x.flatten(1)       # (B, feat_dim)
        return self.classifier(x)  # (B, num_classes)  raw logits


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for bb in ("efficientnet_b3", "resnet50"):
        model = MultiLabelClassifier(backbone=bb, num_classes=20, pretrained=False)
        model.freeze_backbone()
        frozen = model.trainable_params()
        model.unfreeze_backbone()
        full  = model.trainable_params()
        out   = model(torch.randn(2, 3, 320, 320))
        print(f"[{bb}]  frozen={frozen:,}  full={full:,}  out={out.shape}")
