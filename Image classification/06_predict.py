"""Step 6: predict the test set for one classification experiment."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
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
load_test_df = _ds.load_test_df
get_val_transform = _ds.get_val_transform
MultiLabelClassifier = _mdl.MultiLabelClassifier


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        choices=sorted(EXPERIMENTS),
        default=None,
        help="Experiment config to predict with.",
    )
    parser.add_argument(
        "--ckpt",
        choices=["auto", "best", "final", "last"],
        default="auto",
        help="Checkpoint from the experiment folder. auto prefers final, then best.",
    )
    return parser.parse_args()


def resolve_checkpoint(config: ExperimentConfig, ckpt: str) -> Path:
    if ckpt == "auto":
        return config.final_checkpoint if config.final_checkpoint.exists() else config.best_checkpoint
    if ckpt == "final":
        return config.final_checkpoint
    if ckpt == "last":
        return config.last_checkpoint
    return config.best_checkpoint


def predict_tta(model: torch.nn.Module, imgs: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        p1 = torch.sigmoid(model(imgs))
        p2 = torch.sigmoid(model(imgs.flip(-1)))
    return (p1 + p2) / 2


def rle_encode(arr: np.ndarray) -> str:
    pixels = arr.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def load_thresholds(config: ExperimentConfig) -> np.ndarray:
    if config.thresholds_path.exists():
        thresholds = np.load(config.thresholds_path)
        if len(thresholds) != len(LABELS):
            raise ValueError(
                f"Threshold length mismatch in {config.thresholds_path}: "
                f"{len(thresholds)} != {len(LABELS)}"
            )
        print(f"Loaded per-class thresholds from {config.thresholds_path}")
        return thresholds

    print(f"Using default threshold {config.threshold:.2f} for all classes.")
    return np.full(len(LABELS), config.threshold)


def main(
    config: ExperimentConfig | None = None,
    ckpt_path: Path | None = None,
    ckpt_name: str = "auto",
):
    if config is None:
        args = parse_args()
        config = get_experiment_config(args.experiment)
        ckpt_name = args.ckpt

    ensure_experiment_dirs(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = Path(ckpt_path) if ckpt_path else resolve_checkpoint(config, ckpt_name)

    if not path.exists():
        print(f"Checkpoint not found: {path}")
        print("Run 04_train.py first.")
        return None

    print(f"Device: {device}")
    print(f"Experiment: {describe_experiment(config)}")
    print(f"Checkpoint: {path}")
    thresholds = load_thresholds(config)

    model = MultiLabelClassifier(
        backbone=config.backbone,
        num_classes=len(LABELS),
        pretrained=False,
    ).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()

    test_df = load_test_df(DATA_DIR)
    test_ds = VOCDataset(
        test_df,
        DATA_DIR,
        split="test",
        transform=get_val_transform(config.img_size),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )

    all_probs = []
    for imgs, _ in tqdm(test_loader, desc="Predicting (TTA)"):
        probs = predict_tta(model, imgs.to(device)).cpu().numpy()
        all_probs.append(probs)

    all_probs = np.vstack(all_probs)
    all_indices = list(test_ds.indices)
    if all_probs.shape[0] != len(all_indices):
        raise ValueError(
            f"Prediction count mismatch: got {all_probs.shape[0]} rows of probabilities "
            f"for {len(all_indices)} test indices."
        )
    preds = (all_probs > thresholds[None, :]).astype(int)

    prob_df = pd.DataFrame(all_probs, columns=LABELS, index=all_indices)
    prob_df.index.name = "Id"
    pred_df = pd.DataFrame(preds, columns=LABELS, index=all_indices)
    pred_df.index.name = "Id"
    prob_df.to_csv(config.probability_csv_path)
    pred_df.to_csv(config.binary_prediction_csv_path)

    rows = {"Id": [], "Predicted": []}
    for idx in pred_df.index:
        label_vec = pred_df.loc[idx, LABELS].values.astype(int)
        rows["Id"].append(f"{idx}_classification")
        rows["Predicted"].append(rle_encode(label_vec))
        rows["Id"].append(f"{idx}_segmentation")
        rows["Predicted"].append("")

    submission_df = pd.DataFrame(rows).set_index("Id")
    submission_df.to_csv(config.submission_path)

    print(f"\nSubmission saved to {config.submission_path}")
    print(f"Probabilities saved to {config.probability_csv_path}")
    print(f"Binary predictions saved to {config.binary_prediction_csv_path}")
    print(f"Rows: {len(submission_df)} (classification + empty segmentation)")
    print("\nFirst 4 rows:")
    print(submission_df.head(4))
    return pred_df


if __name__ == "__main__":
    main()
