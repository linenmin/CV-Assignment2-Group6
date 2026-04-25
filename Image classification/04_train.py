"""Step 4: train one configured image-classification experiment."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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
    build_criterion,
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
get_train_transform = _ds.get_train_transform
get_val_transform = _ds.get_val_transform
MultiLabelClassifier = _mdl.MultiLabelClassifier


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        choices=sorted(EXPERIMENTS),
        default=None,
        help="Experiment config to train.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loaders(data_dir: Path, config: ExperimentConfig, device: torch.device):
    df = load_train_df(data_dir)
    train_idx, val_idx = train_test_split(
        range(len(df)),
        test_size=config.val_split,
        random_state=config.random_seed,
    )

    train_ds = VOCDataset(
        df.iloc[train_idx],
        data_dir,
        split="train",
        transform=get_train_transform(config.img_size),
    )
    val_ds = VOCDataset(
        df.iloc[val_idx],
        data_dir,
        split="train",
        transform=get_val_transform(config.img_size),
    )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def run_epoch(model, loader, criterion, optimizer, train: bool, device: torch.device) -> float:
    model.train(train)
    total_loss = 0.0
    desc = "train" if train else "val  "

    with torch.set_grad_enabled(train):
        for imgs, labels in tqdm(loader, leave=False, desc=desc):
            imgs = imgs.to(device)
            labels = labels.to(device)

            logits = model(imgs)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(imgs)

    return total_loss / len(loader.dataset)


def train_stage(
    model,
    train_loader,
    val_loader,
    criterion,
    config: ExperimentConfig,
    device: torch.device,
    lr: float,
    epochs: int,
    stage_name: str,
    best_val_loss: float,
    history: list[dict],
) -> float:
    optimizer = torch.optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, True, device)
        val_loss = run_epoch(model, val_loader, criterion, optimizer, False, device)
        scheduler.step()

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.best_checkpoint)

        torch.save(model.state_dict(), config.last_checkpoint)
        history.append(
            {
                "experiment": config.name,
                "stage": stage_name,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": lr,
                "is_best": is_best,
            }
        )

        flag = "  <- best" if is_best else ""
        print(
            f"[{stage_name}] Epoch {epoch:3d}/{epochs} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}{flag}"
        )

    return best_val_loss


def main(config: ExperimentConfig | None = None):
    if config is None:
        args = parse_args()
        config = get_experiment_config(args.experiment)

    ensure_experiment_dirs(config)
    set_seed(config.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Experiment: {describe_experiment(config)}")
    print(f"Outputs: {config.output_dir}")

    train_loader, val_loader = build_loaders(DATA_DIR, config, device)
    model = MultiLabelClassifier(
        backbone=config.backbone,
        num_classes=len(LABELS),
        pretrained=True,
    ).to(device)
    criterion = build_criterion(config)
    history = []
    best_val_loss = float("inf")

    print(f"\n=== Stage 1: train head only ({config.backbone}, frozen) ===")
    model.freeze_backbone()
    print(f"Trainable params: {model.trainable_params():,}")
    best_val_loss = train_stage(
        model,
        train_loader,
        val_loader,
        criterion,
        config,
        device,
        lr=config.stage1_lr,
        epochs=config.stage1_epochs,
        stage_name="S1",
        best_val_loss=best_val_loss,
        history=history,
    )

    print("\n=== Stage 2: full fine-tuning ===")
    model.unfreeze_backbone()
    print(f"Trainable params: {model.trainable_params():,}")
    best_val_loss = train_stage(
        model,
        train_loader,
        val_loader,
        criterion,
        config,
        device,
        lr=config.stage2_lr,
        epochs=config.stage2_epochs,
        stage_name="S2",
        best_val_loss=best_val_loss,
        history=history,
    )

    print("\n=== Stage 3: retrain on all training images ===")
    model.load_state_dict(torch.load(config.best_checkpoint, map_location=device))
    full_df = load_train_df(DATA_DIR)
    full_ds = VOCDataset(
        full_df,
        DATA_DIR,
        split="train",
        transform=get_train_transform(config.img_size),
    )
    full_loader = DataLoader(
        full_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.stage3_lr,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.stage3_epochs,
    )
    for epoch in range(1, config.stage3_epochs + 1):
        loss = run_epoch(model, full_loader, criterion, optimizer, True, device)
        scheduler.step()
        history.append(
            {
                "experiment": config.name,
                "stage": "S3",
                "epoch": epoch,
                "train_loss": loss,
                "val_loss": np.nan,
                "lr": config.stage3_lr,
                "is_best": False,
            }
        )
        print(f"[S3] Epoch {epoch:3d}/{config.stage3_epochs} | full_train_loss={loss:.4f}")

    torch.save(model.state_dict(), config.final_checkpoint)
    pd.DataFrame(history).to_csv(config.training_history_path, index=False)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints: {config.checkpoints_dir}")
    print(f"Training history: {config.training_history_path}")
    return config


if __name__ == "__main__":
    main()
