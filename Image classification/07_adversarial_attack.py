"""Step 7: simple white-box adversarial attacks on the classifier.

This script evaluates targeted FGSM/PGD attacks on the validation split. The
default target is to make the classifier predict "aeroplane" for each attacked
image while keeping the perturbation bounded in normalized image space.
"""

import argparse
import sys
from pathlib import Path

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
    return parser.parse_args()


def resolve_checkpoint(config: ExperimentConfig, ckpt: str) -> Path:
    return config.final_checkpoint if ckpt == "final" else config.best_checkpoint


def load_val_subset(config: ExperimentConfig, max_samples: int) -> DataLoader:
    df = load_train_df(DATA_DIR)
    _, val_idx = train_test_split(
        range(len(df)),
        test_size=config.val_split,
        random_state=config.random_seed,
    )
    val_df = df.iloc[val_idx[:max_samples]]
    val_ds = VOCDataset(
        val_df,
        DATA_DIR,
        split="train",
        transform=get_val_transform(config.img_size),
    )
    return DataLoader(
        val_ds,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )


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
    return adv.detach()


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
        adv = (clean + delta).detach()
    return adv


def summarize_probs(
    clean_probs: np.ndarray,
    adv_probs: np.ndarray,
    target_index: int,
    threshold: float,
) -> dict:
    clean_target = clean_probs[:, target_index]
    adv_target = adv_probs[:, target_index]
    return {
        "clean_target_prob_mean": float(clean_target.mean()),
        "adv_target_prob_mean": float(adv_target.mean()),
        "target_activation_rate_clean": float((clean_target > threshold).mean()),
        "target_activation_rate_adv": float((adv_target > threshold).mean()),
    }


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

    threshold = config.threshold
    if config.thresholds_path.exists():
        threshold = float(np.load(config.thresholds_path)[target_index])

    loader = load_val_subset(config, args.max_samples)
    clean_chunks = []
    fgsm_chunks = []
    pgd_chunks = []

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

    clean_probs = np.vstack(clean_chunks)
    fgsm_probs = np.vstack(fgsm_chunks)
    pgd_probs = np.vstack(pgd_chunks)
    rows = [
        {
            "attack": "FGSM",
            "target_class": args.target_class,
            "epsilon": args.epsilon,
            "alpha": np.nan,
            "steps": 1,
            **summarize_probs(clean_probs, fgsm_probs, target_index, threshold),
        },
        {
            "attack": "PGD",
            "target_class": args.target_class,
            "epsilon": args.epsilon,
            "alpha": args.alpha,
            "steps": args.pgd_steps,
            **summarize_probs(clean_probs, pgd_probs, target_index, threshold),
        },
    ]
    out_path = config.metrics_dir / f"adversarial_{args.target_class}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Saved adversarial summary: {out_path}")


if __name__ == "__main__":
    main()
