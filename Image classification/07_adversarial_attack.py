"""Step 7: simple white-box adversarial attacks on the classifier.

This script evaluates targeted FGSM/PGD attacks on the validation split. The
default target is to make the classifier predict "aeroplane" for each attacked
image while keeping the perturbation bounded in normalized image space.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from shared import (  # noqa: E402
    DATA_DIR,
    EXPERIMENTS,
    LABELS,
    NUM_WORKERS,
    ExperimentConfig,
    describe_experiment,
    ensure_experiment_dirs,
    get_experiment_config,
    load_module,
)

_here = Path(__file__).parent
_ds = load_module("dataset", _here / "02_dataset.py")
_mdl = load_module("model", _here / "03_model.py")

VOCDataset = _ds.VOCDataset
load_train_df = _ds.load_train_df
get_val_transform = _ds.get_val_transform
MultiLabelClassifier = _mdl.MultiLabelClassifier

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), default=None)
    parser.add_argument("--target-class", default="aeroplane", choices=LABELS)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--pgd-steps", type=int, default=10)
    parser.add_argument(
        "--ckpt",
        choices=["best", "final"],
        default="final",
        help="Use final checkpoint by default because it is used for submissions.",
    )
    parser.add_argument(
        "--target-absent-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate only validation images whose ground-truth target label is 0.",
    )
    return parser.parse_args()


def resolve_checkpoint(config: ExperimentConfig, ckpt: str) -> Path:
    return config.final_checkpoint if ckpt == "final" else config.best_checkpoint


def load_val_subset(
    config: ExperimentConfig,
    max_samples: int,
    target_class: str,
    target_absent_only: bool,
) -> DataLoader:
    df = load_train_df(DATA_DIR)
    _, val_idx = train_test_split(
        range(len(df)),
        test_size=config.val_split,
        random_state=config.random_seed,
    )
    val_df = df.iloc[val_idx].copy()
    if target_absent_only:
        val_df = val_df[val_df[target_class] == 0]
    val_df = val_df.iloc[:max_samples]
    if val_df.empty:
        raise ValueError(
            f"No validation samples available for target_class={target_class!r} "
            f"with target_absent_only={target_absent_only}."
        )
    val_ds = VOCDataset(
        val_df,
        DATA_DIR,
        split="train",
        transform=get_val_transform(config.img_size, mode=config.transform_mode),
    )
    return DataLoader(
        val_ds,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )


def clamp_normalized(imgs: torch.Tensor) -> torch.Tensor:
    mean = MEAN.to(imgs.device)
    std = STD.to(imgs.device)
    lower = (0.0 - mean) / std
    upper = (1.0 - mean) / std
    return torch.max(torch.min(imgs, upper), lower)


def denormalize(imgs: torch.Tensor) -> torch.Tensor:
    mean = MEAN.to(imgs.device)
    std = STD.to(imgs.device)
    return (imgs * std + mean).clamp(0, 1)


def targeted_loss(logits: torch.Tensor, target_index: int) -> torch.Tensor:
    targets = torch.zeros_like(logits)
    targets[:, target_index] = 1.0
    return F.binary_cross_entropy_with_logits(logits, targets)


def fgsm_targeted(
    model: torch.nn.Module,
    imgs: torch.Tensor,
    target_index: int,
    epsilon: float,
) -> torch.Tensor:
    adv = imgs.detach().clone().requires_grad_(True)
    loss = targeted_loss(model(adv), target_index)
    grad = torch.autograd.grad(loss, adv)[0]
    # Targeted attack: minimize the target loss.
    adv = adv - epsilon * grad.sign()
    return clamp_normalized(adv).detach()


def pgd_targeted(
    model: torch.nn.Module,
    imgs: torch.Tensor,
    target_index: int,
    epsilon: float,
    alpha: float,
    steps: int,
) -> torch.Tensor:
    clean = imgs.detach()
    adv = clean.clone()
    for _ in range(steps):
        adv.requires_grad_(True)
        loss = targeted_loss(model(adv), target_index)
        grad = torch.autograd.grad(loss, adv)[0]
        adv = adv - alpha * grad.sign()
        delta = (adv - clean).clamp(min=-epsilon, max=epsilon)
        adv = clamp_normalized(clean + delta).detach()
    return adv


def summarize_probs(
    clean_probs: np.ndarray,
    adv_probs: np.ndarray,
    clean_imgs: torch.Tensor,
    adv_imgs: torch.Tensor,
    target_index: int,
    threshold: float,
) -> dict:
    clean_target = clean_probs[:, target_index]
    adv_target = adv_probs[:, target_index]
    normalized_delta = adv_imgs - clean_imgs
    pixel_delta = denormalize(adv_imgs) - denormalize(clean_imgs)
    return {
        "num_samples": int(clean_probs.shape[0]),
        "clean_target_prob_mean": float(clean_target.mean()),
        "adv_target_prob_mean": float(adv_target.mean()),
        "target_activation_rate_clean": float((clean_target > threshold).mean()),
        "target_activation_rate_adv": float((adv_target > threshold).mean()),
        "attack_success_rate": float((adv_target > threshold).mean()),
        "normalized_linf_mean": float(
            normalized_delta.abs().flatten(1).amax(dim=1).mean().item()
        ),
        "normalized_l2_mean": float(
            normalized_delta.flatten(1).norm(p=2, dim=1).mean().item()
        ),
        "pixel_linf_mean": float(
            pixel_delta.abs().flatten(1).amax(dim=1).mean().item()
        ),
        "pixel_l2_mean": float(pixel_delta.flatten(1).norm(p=2, dim=1).mean().item()),
    }


def plot_attack_examples(
    clean_imgs: torch.Tensor,
    adv_imgs: torch.Tensor,
    clean_target_probs: np.ndarray,
    adv_target_probs: np.ndarray,
    out_path: Path,
    max_examples: int = 4,
) -> None:
    n = min(max_examples, clean_imgs.shape[0])
    clean_vis = denormalize(clean_imgs[:n]).cpu()
    adv_vis = denormalize(adv_imgs[:n]).cpu()
    perturb_vis = (0.5 + 10.0 * (adv_vis - clean_vis)).clamp(0, 1)

    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for row in range(n):
        images = (clean_vis[row], perturb_vis[row], adv_vis[row])
        for col, image in enumerate(images):
            axes[row, col].imshow(image.permute(1, 2, 0).numpy())
            axes[row, col].axis("off")
        axes[row, 0].set_title(f"Clean image\np={clean_target_probs[row]:.3f}")
        axes[row, 1].set_title("Perturbation x10")
        axes[row, 2].set_title(f"Adversarial image\np={adv_target_probs[row]:.3f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    args = parse_args()
    config = get_experiment_config(args.experiment)
    ensure_experiment_dirs(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_index = LABELS.index(args.target_class)
    checkpoint = resolve_checkpoint(config, args.ckpt)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    print(f"Device: {device}")
    print(f"Experiment: {describe_experiment(config)}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Target class: {args.target_class}")

    model = MultiLabelClassifier(
        backbone=config.backbone,
        num_classes=len(LABELS),
        pretrained=False,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    threshold = config.threshold
    if config.thresholds_path.exists():
        threshold = float(np.load(config.thresholds_path)[target_index])

    loader = load_val_subset(
        config,
        args.max_samples,
        args.target_class,
        args.target_absent_only,
    )
    print(f"Samples: {len(loader.dataset)}")
    print(f"Target threshold: {threshold:.3f}")
    clean_chunks = []
    fgsm_chunks = []
    pgd_chunks = []
    clean_img_chunks = []
    fgsm_img_chunks = []
    pgd_img_chunks = []

    for imgs, _ in tqdm(loader, desc="Attacking validation images"):
        imgs = imgs.to(device)
        clean = torch.sigmoid(model(imgs)).detach().cpu().numpy()
        fgsm = fgsm_targeted(model, imgs, target_index, args.epsilon)
        pgd = pgd_targeted(model, imgs, target_index, args.epsilon, args.alpha, args.pgd_steps)
        fgsm_probs = torch.sigmoid(model(fgsm)).detach().cpu().numpy()
        pgd_probs = torch.sigmoid(model(pgd)).detach().cpu().numpy()
        clean_chunks.append(clean)
        fgsm_chunks.append(fgsm_probs)
        pgd_chunks.append(pgd_probs)
        clean_img_chunks.append(imgs.detach().cpu())
        fgsm_img_chunks.append(fgsm.detach().cpu())
        pgd_img_chunks.append(pgd.detach().cpu())

    clean_probs = np.vstack(clean_chunks)
    fgsm_probs = np.vstack(fgsm_chunks)
    pgd_probs = np.vstack(pgd_chunks)
    clean_imgs = torch.cat(clean_img_chunks)
    fgsm_imgs = torch.cat(fgsm_img_chunks)
    pgd_imgs = torch.cat(pgd_img_chunks)
    rows = [
        {
            "attack": "FGSM",
            "target_class": args.target_class,
            "epsilon": args.epsilon,
            "alpha": np.nan,
            "steps": 1,
            **summarize_probs(
                clean_probs,
                fgsm_probs,
                clean_imgs,
                fgsm_imgs,
                target_index,
                threshold,
            ),
        },
        {
            "attack": "PGD",
            "target_class": args.target_class,
            "epsilon": args.epsilon,
            "alpha": args.alpha,
            "steps": args.pgd_steps,
            **summarize_probs(
                clean_probs,
                pgd_probs,
                clean_imgs,
                pgd_imgs,
                target_index,
                threshold,
            ),
        },
    ]
    out_path = config.metrics_dir / f"adversarial_{args.target_class}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    fig_path = config.figures_dir / f"adversarial_{args.target_class}_examples.png"
    plot_attack_examples(
        clean_imgs,
        pgd_imgs,
        clean_probs[:, target_index],
        pgd_probs[:, target_index],
        fig_path,
    )
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Saved adversarial summary: {out_path}")
    print(f"Saved adversarial examples: {fig_path}")


if __name__ == "__main__":
    main()
