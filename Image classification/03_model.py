"""Step 3: swappable-backbone multi-label classifier."""

import torch
import torch.nn as nn
import torchvision.models as models

_BACKBONE_CONFIGS = {
    "resnet50": {
        "factory": lambda pretrained: models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        ),
        "feat_dim": 2048,
        "features_fn": lambda model: nn.Sequential(*list(model.children())[:-1]),
    },
    "efficientnet_b3": {
        "factory": lambda pretrained: models.efficientnet_b3(
            weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        ),
        "feat_dim": 1536,
        "features_fn": lambda model: nn.Sequential(model.features, model.avgpool),
    },
    "convnext_tiny": {
        "factory": lambda pretrained: models.convnext_tiny(
            weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        ),
        "feat_dim": 768,
        "features_fn": lambda model: nn.Sequential(model.features, model.avgpool),
    },
    "convnext_small": {
        "factory": lambda pretrained: models.convnext_small(
            weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
        ),
        "feat_dim": 768,
        "features_fn": lambda model: nn.Sequential(model.features, model.avgpool),
    },
    "convnext_base": {
        "factory": lambda pretrained: models.convnext_base(
            weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
        ),
        "feat_dim": 1024,
        "features_fn": lambda model: nn.Sequential(model.features, model.avgpool),
    },
    "convnext_large": {
        "factory": lambda pretrained: models.convnext_large(
            weights=models.ConvNeXt_Large_Weights.IMAGENET1K_V1 if pretrained else None
        ),
        "feat_dim": 1536,
        "features_fn": lambda model: nn.Sequential(model.features, model.avgpool),
    },
    "efficientnet_v2_s": {
        "factory": lambda pretrained: models.efficientnet_v2_s(
            weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        ),
        "feat_dim": 1280,
        "features_fn": lambda model: nn.Sequential(model.features, model.avgpool),
    },
}


class MultiLabelClassifier(nn.Module):
    """ImageNet backbone plus a small multi-label classification head."""

    def __init__(
        self,
        backbone: str = "efficientnet_b3",
        num_classes: int = 20,
        pretrained: bool = True,
    ):
        super().__init__()
        if backbone not in _BACKBONE_CONFIGS:
            valid = ", ".join(sorted(_BACKBONE_CONFIGS))
            raise ValueError(f"Unknown backbone '{backbone}'. Valid: {valid}")

        cfg = _BACKBONE_CONFIGS[backbone]
        model = cfg["factory"](pretrained)
        self.backbone = backbone
        self.features = cfg["features_fn"](model)
        self._feat_dim = cfg["feat_dim"]

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
        for parameter in self.features.parameters():
            parameter.requires_grad = False

    def unfreeze_backbone(self):
        for parameter in self.features.parameters():
            parameter.requires_grad = True

    def trainable_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


if __name__ == "__main__":
    for backbone in sorted(_BACKBONE_CONFIGS):
        model = MultiLabelClassifier(backbone=backbone, num_classes=20, pretrained=False)
        model.freeze_backbone()
        frozen = model.trainable_params()
        model.unfreeze_backbone()
        full = model.trainable_params()
        out = model(torch.randn(2, 3, 320, 320))
        print(f"[{backbone}] frozen={frozen:,} full={full:,} out={out.shape}")
