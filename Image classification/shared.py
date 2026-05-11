"""Shared constants, experiment configs, utilities, and losses."""

from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path

import torch
import torch.nn as nn

# Windows uses "spawn" multiprocessing: worker processes cannot re-import
# modules loaded dynamically via importlib, so num_workers must be 0.
NUM_WORKERS = 0 if os.name == "nt" else 4

# Class order must match train_set.csv columns and Kaggle submission vectors.
LABELS = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "kul-computer-vision-ga-2-2026"
OUTPUT_DIR = PROJECT_ROOT / "output"
CLASSIFICATION_OUTPUT_DIR = OUTPUT_DIR / "image_classification"
DATA_EXPORT_DIR = CLASSIFICATION_OUTPUT_DIR / "_data"


@dataclass(frozen=True)
class ExperimentConfig:
    """Settings and output paths that must stay together for one model."""

    name: str
    backbone: str
    img_size: int
    batch_size: int
    eval_batch_size: int
    stage1_epochs: int = 5
    stage1_lr: float = 1e-3
    stage2_epochs: int = 20
    stage2_lr: float = 1e-4
    stage3_epochs: int = 5
    stage3_lr: float = 5e-5
    weight_decay: float = 1e-4
    val_split: float = 0.2
    random_seed: int = 42
    loss_name: str = "asymmetric"
    threshold: float = 0.5
    transform_mode: str = "resize"
    sampler: str = "none"
    early_stop_patience: int | None = None

    @property
    def output_dir(self) -> Path:
        return CLASSIFICATION_OUTPUT_DIR / self.name

    @property
    def checkpoints_dir(self) -> Path:
        return self.output_dir / "checkpoints"

    @property
    def metrics_dir(self) -> Path:
        return self.output_dir / "metrics"

    @property
    def figures_dir(self) -> Path:
        return self.output_dir / "figures"

    @property
    def submissions_dir(self) -> Path:
        return self.output_dir / "submissions"

    @property
    def predictions_dir(self) -> Path:
        return self.output_dir / "predictions"

    @property
    def best_checkpoint(self) -> Path:
        return self.checkpoints_dir / "best_model.pth"

    @property
    def final_checkpoint(self) -> Path:
        return self.checkpoints_dir / "final_model.pth"

    @property
    def last_checkpoint(self) -> Path:
        return self.checkpoints_dir / "last_model.pth"

    @property
    def thresholds_path(self) -> Path:
        return self.metrics_dir / "best_thresholds.npy"

    @property
    def ap_csv_path(self) -> Path:
        return self.metrics_dir / "ap_per_class.csv"

    @property
    def training_history_path(self) -> Path:
        return self.metrics_dir / "training_history.csv"

    @property
    def training_history_plot_path(self) -> Path:
        return self.figures_dir / "training_history.png"

    @property
    def eval_summary_path(self) -> Path:
        return self.metrics_dir / "evaluation_summary.csv"

    @property
    def ap_plot_path(self) -> Path:
        return self.figures_dir / "eval_ap_per_class.png"

    @property
    def submission_path(self) -> Path:
        return self.submissions_dir / f"submission_classification_{self.name}.csv"

    @property
    def probability_csv_path(self) -> Path:
        return self.predictions_dir / f"test_probabilities_{self.name}.csv"

    @property
    def binary_prediction_csv_path(self) -> Path:
        return self.predictions_dir / f"test_binary_predictions_{self.name}.csv"


DEFAULT_EXPERIMENT = "convnext_small_320"

EXPERIMENTS = {
    "efficientnet_b3_320": ExperimentConfig(
        name="efficientnet_b3_320",
        backbone="efficientnet_b3",
        img_size=320,
        batch_size=16,
        eval_batch_size=32,
    ),
    "convnext_tiny_320": ExperimentConfig(
        name="convnext_tiny_320",
        backbone="convnext_tiny",
        img_size=320,
        batch_size=16,
        eval_batch_size=32,
    ),
    "convnext_small_320": ExperimentConfig(
        name="convnext_small_320",
        backbone="convnext_small",
        img_size=320,
        batch_size=8,
        eval_batch_size=16,
    ),
    "convnext_small_320_pad_sampler": ExperimentConfig(
        name="convnext_small_320_pad_sampler",
        backbone="convnext_small",
        img_size=320,
        batch_size=8,
        eval_batch_size=16,
        transform_mode="square_pad",
        sampler="weak_class_weighted",
        early_stop_patience=4,
    ),
    "convnext_base_320": ExperimentConfig(
        name="convnext_base_320",
        backbone="convnext_base",
        img_size=320,
        batch_size=4,
        eval_batch_size=8,
        early_stop_patience=4,
    ),
    "convnext_large_320": ExperimentConfig(
        name="convnext_large_320",
        backbone="convnext_large",
        img_size=320,
        batch_size=2,
        eval_batch_size=4,
    ),
    "convnextv2_tiny_320": ExperimentConfig(
        name="convnextv2_tiny_320",
        backbone="convnextv2_tiny",
        img_size=320,
        batch_size=8,
        eval_batch_size=16,
        early_stop_patience=4,
    ),
    "convnextv2_base_320": ExperimentConfig(
        name="convnextv2_base_320",
        backbone="convnextv2_base",
        img_size=320,
        batch_size=4,
        eval_batch_size=8,
        early_stop_patience=4,
    ),
    "vit_b_16_224": ExperimentConfig(
        name="vit_b_16_224",
        backbone="vit_b_16",
        img_size=224,
        batch_size=8,
        eval_batch_size=16,
        early_stop_patience=4,
    ),
    "vit_l_16_224": ExperimentConfig(
        name="vit_l_16_224",
        backbone="vit_l_16",
        img_size=224,
        batch_size=2,
        eval_batch_size=4,
        early_stop_patience=4,
    ),
    "resnet50_224": ExperimentConfig(
        name="resnet50_224",
        backbone="resnet50",
        img_size=224,
        batch_size=32,
        eval_batch_size=32,
    ),
    "efficientnet_v2_s_320": ExperimentConfig(
        name="efficientnet_v2_s_320",
        backbone="efficientnet_v2_s",
        img_size=320,
        batch_size=16,
        eval_batch_size=32,
        early_stop_patience=5,
    ),
}


def get_experiment_config(name: str | None = None) -> ExperimentConfig:
    experiment_name = name or DEFAULT_EXPERIMENT
    try:
        return EXPERIMENTS[experiment_name]
    except KeyError as exc:
        valid = ", ".join(sorted(EXPERIMENTS))
        raise ValueError(f"Unknown experiment '{experiment_name}'. Valid: {valid}") from exc


def ensure_experiment_dirs(config: ExperimentConfig) -> None:
    for path in (
        config.checkpoints_dir,
        config.metrics_dir,
        config.figures_dir,
        config.submissions_dir,
        config.predictions_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)


def describe_experiment(config: ExperimentConfig) -> str:
    return (
        f"{config.name} | backbone={config.backbone} | "
        f"img_size={config.img_size} | batch_size={config.batch_size}"
    )


def load_module(name: str, filepath: Path):
    """Load local numbered scripts such as 02_dataset.py."""

    spec = importlib.util.spec_from_file_location(name, filepath)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {name} from {filepath}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class AsymmetricLoss(nn.Module):
    """Asymmetric loss for noisy multi-label classification.

    Based on Ridnik et al., "Asymmetric Loss For Multi-Label Classification".
    The negative clipping and focal decay reduce the penalty from easy or
    potentially unlabelled negatives, which is helpful for PASCAL VOC.
    """

    def __init__(
        self,
        gamma_neg: float = 4,
        gamma_pos: float = 0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_neg = 1.0 - probs

        if self.clip > 0:
            probs_neg = (probs_neg + self.clip).clamp(max=1.0)

        log_p = torch.log(probs.clamp(min=self.eps))
        log_np = torch.log(probs_neg.clamp(min=self.eps))
        loss = targets * log_p + (1 - targets) * log_np

        with torch.no_grad():
            weights = (
                targets * (1 - probs).pow(self.gamma_pos)
                + (1 - targets) * probs.pow(self.gamma_neg)
            )
        loss = loss * weights

        return -loss.mean()


def build_criterion(config: ExperimentConfig) -> nn.Module:
    if config.loss_name == "asymmetric":
        return AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05)
    if config.loss_name == "bce":
        return nn.BCEWithLogitsLoss()
    raise ValueError(f"Unknown loss_name: {config.loss_name}")
