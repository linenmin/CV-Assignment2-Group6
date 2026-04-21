import torch
import torch.nn.functional as F
from mmseg.models.decode_heads import LightHamHead
from mmseg.models.utils import resize
from mmseg.registry import MODELS


@MODELS.register_module()
class RegionRebalanceLightHamHead(LightHamHead):
    """SegNeXt decode head with a training-only region rebalance branch.

    The auxiliary branch pools per-class region features from the aligned
    decode features and applies a balanced-softmax style classification loss.
    Inference still uses the standard pixel classifier only.
    """

    def __init__(
        self,
        region_class_frequencies: list[int],
        region_loss_weight: float = 0.5,
        include_background_in_region_branch: bool = False,
        region_min_pixels: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.region_loss_weight = float(region_loss_weight)
        self.include_background_in_region_branch = bool(include_background_in_region_branch)
        self.region_min_pixels = int(region_min_pixels)

        region_classes = self.num_classes if self.include_background_in_region_branch else self.num_classes - 1
        if len(region_class_frequencies) != region_classes:
            raise ValueError(
                "region_class_frequencies must match the number of classes used by the region branch"
            )

        self.region_classifier = torch.nn.Linear(self.channels, region_classes)
        region_freq_tensor = torch.tensor(region_class_frequencies, dtype=torch.float32).clamp_min(1.0)
        self.register_buffer("region_log_frequencies", region_freq_tensor.log(), persistent=False)

    def _forward_feature(self, inputs):
        inputs = self._transform_inputs(inputs)
        inputs = [
            resize(
                level,
                size=inputs[0].shape[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
            for level in inputs
        ]

        x = torch.cat(inputs, dim=1)
        x = self.squeeze(x)
        x = self.hamburger(x)
        return self.align(x)

    def forward(self, inputs):
        feature = self._forward_feature(inputs)
        return self.cls_seg(feature)

    def _collect_region_features(self, feature_map: torch.Tensor, seg_label: torch.Tensor):
        region_features: list[torch.Tensor] = []
        region_targets: list[int] = []

        feature_map = feature_map.permute(0, 2, 3, 1)
        for batch_index in range(seg_label.shape[0]):
            labels = torch.unique(seg_label[batch_index])
            for label in labels.tolist():
                if label == self.ignore_index:
                    continue
                if not self.include_background_in_region_branch and label == 0:
                    continue

                mask = seg_label[batch_index] == label
                if int(mask.sum().item()) < self.region_min_pixels:
                    continue

                region_feature = feature_map[batch_index][mask].mean(dim=0)
                region_features.append(region_feature)
                region_targets.append(label if self.include_background_in_region_branch else label - 1)

        return region_features, region_targets

    def region_loss_by_feat(self, feature_map: torch.Tensor, batch_data_samples) -> torch.Tensor | None:
        seg_label = self._stack_batch_gt(batch_data_samples).squeeze(1)
        feature_map = resize(
            input=feature_map,
            size=seg_label.shape[1:],
            mode="bilinear",
            align_corners=self.align_corners,
        )

        region_features, region_targets = self._collect_region_features(feature_map, seg_label)
        if not region_features:
            return None

        region_features_tensor = torch.stack(region_features)
        region_targets_tensor = torch.tensor(
            region_targets,
            device=region_features_tensor.device,
            dtype=torch.long,
        )

        region_logits = self.region_classifier(region_features_tensor)
        adjusted_logits = region_logits + self.region_log_frequencies.to(region_logits.device).unsqueeze(0)
        return F.cross_entropy(adjusted_logits, region_targets_tensor)

    def loss(self, inputs, batch_data_samples, train_cfg):
        feature = self._forward_feature(inputs)
        seg_logits = self.cls_seg(feature)
        losses = self.loss_by_feat(seg_logits, batch_data_samples)

        region_loss = self.region_loss_by_feat(feature, batch_data_samples)
        if region_loss is not None and self.region_loss_weight > 0:
            losses["loss_region_rebalance"] = region_loss * self.region_loss_weight
        return losses
