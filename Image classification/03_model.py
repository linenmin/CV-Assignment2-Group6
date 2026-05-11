"""Step 3: swappable-backbone multi-label classifier."""

import torch
import torch.nn as nn
import torchvision.models as models


_CONVNEXTV2_VARIANTS = {
    "tiny": {
        "model_id": "facebook/convnextv2-tiny-1k-224",
        "hidden_sizes": (96, 192, 384, 768),
        "depths": (3, 3, 9, 3),
        "feat_dim": 768,
    },
    "base": {
        "model_id": "facebook/convnextv2-base-1k-224",
        "hidden_sizes": (128, 256, 512, 1024),
        "depths": (3, 3, 27, 3),
        "feat_dim": 1024,
    },
}


class _ViTFeatures(nn.Module):
    """Feature extractor that returns the ViT class token before its head."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.image_size = model.image_size
        self.patch_size = model.patch_size
        self.hidden_dim = model.hidden_dim
        self.conv_proj = model.conv_proj
        self.class_token = model.class_token
        self.encoder = model.encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, _, h, w = x.shape
        torch._assert(
            h == self.image_size,
            f"Wrong image height: expected {self.image_size}, got {h}",
        )
        torch._assert(
            w == self.image_size,
            f"Wrong image width: expected {self.image_size}, got {w}",
        )

        x = self.conv_proj(x)
        x = x.reshape(n, self.hidden_dim, -1)
        x = x.permute(0, 2, 1)

        class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([class_token, x], dim=1)
        x = self.encoder(x)
        return x[:, 0]


class _ConvNextV2Features(nn.Module):
    """Hugging Face ConvNeXt V2 backbone returning pooled image features."""

    def __init__(self, variant: str, pretrained: bool):
        super().__init__()
        cfg = _CONVNEXTV2_VARIANTS[variant]
        ConvNextV2Config, ConvNextV2Model = _load_convnextv2_classes()

        if pretrained:
            self.model = ConvNextV2Model.from_pretrained(cfg["model_id"])
        else:
            config = ConvNextV2Config(
                image_size=224,
                hidden_sizes=cfg["hidden_sizes"],
                depths=cfg["depths"],
            )
            self.model = ConvNextV2Model(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=x).pooler_output


def _load_convnextv2_classes():
    try:
        import transformers
    except ImportError as exc:
        raise ImportError(
            "ConvNeXt V2 backbones require the optional 'transformers' package. "
            "Install it in the biometrics environment with: "
            'C:\\Users\\31667\\.conda\\envs\\biometrics\\python.exe -m pip install transformers'
        ) from exc

    config_cls = getattr(transformers, "ConvNextV2Config", None) or getattr(
        transformers,
        "ConvNeXTV2Config",
        None,
    )
    model_cls = getattr(transformers, "ConvNextV2Model", None)
    if config_cls is None or model_cls is None:
        raise ImportError(
            "Installed transformers does not expose ConvNextV2Config/ConvNextV2Model. "
            "Upgrade transformers in the biometrics environment."
        )
    return config_cls, model_cls


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
    "convnextv2_tiny": {
        "factory": lambda pretrained: _ConvNextV2Features("tiny", pretrained),
        "feat_dim": _CONVNEXTV2_VARIANTS["tiny"]["feat_dim"],
        "features_fn": lambda model: model,
    },
    "convnextv2_base": {
        "factory": lambda pretrained: _ConvNextV2Features("base", pretrained),
        "feat_dim": _CONVNEXTV2_VARIANTS["base"]["feat_dim"],
        "features_fn": lambda model: model,
    },
    "vit_b_16": {
        "factory": lambda pretrained: models.vit_b_16(
            weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        ),
        "feat_dim": 768,
        "features_fn": lambda model: _ViTFeatures(model),
        "smoke_size": 224,
    },
    "vit_l_16": {
        "factory": lambda pretrained: models.vit_l_16(
            weights=models.ViT_L_16_Weights.IMAGENET1K_V1 if pretrained else None
        ),
        "feat_dim": 1024,
        "features_fn": lambda model: _ViTFeatures(model),
        "smoke_size": 224,
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
        size = _BACKBONE_CONFIGS[backbone].get("smoke_size", 320)
        out = model(torch.randn(2, 3, size, size))
        print(f"[{backbone}] frozen={frozen:,} full={full:,} out={out.shape}")
