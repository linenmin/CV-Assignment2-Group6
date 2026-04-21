"""
Step 3: ResNet-50 multi-label classifier with two-stage fine-tune support.

Importable by train, evaluate, and predict scripts.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet50Classifier(nn.Module):
    """
    ResNet-50 backbone with a custom multi-label classification head.

    Usage
    -----
    model = ResNet50Classifier(num_classes=20, pretrained=True)
    model.freeze_backbone()          # Stage 1: train head only
    ...
    model.unfreeze_backbone()        # Stage 2: full fine-tune
    """

    def __init__(self, num_classes: int = 20, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet50(weights=weights)

        # Remove the final FC layer; keep everything up to the global avg-pool
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # → (B, 2048, 1, 1)

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes),
            # No Sigmoid here — BCEWithLogitsLoss handles it numerically
        )

        # Initialise the new head with sensible weights
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Fine-tune helpers
    # ------------------------------------------------------------------
    def freeze_backbone(self):
        """Freeze all backbone parameters (Stage 1)."""
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters (Stage 2)."""
        for p in self.features.parameters():
            p.requires_grad = True

    def trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)    # (B, 2048, 1, 1)
        x = x.flatten(1)        # (B, 2048)
        return self.classifier(x)  # (B, num_classes)  raw logits


# ---------------------------------------------------------------------------
# Quick sanity check when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = ResNet50Classifier(num_classes=20, pretrained=True)

    model.freeze_backbone()
    print(f"[Stage 1] Trainable params: {model.trainable_params():,}")

    model.unfreeze_backbone()
    print(f"[Stage 2] Trainable params: {model.trainable_params():,}")

    dummy = torch.randn(4, 3, 224, 224)
    logits = model(dummy)
    print(f"Output shape: {logits.shape}")  # (4, 20)
    print("Model OK.")
