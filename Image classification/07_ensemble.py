"""Step 7: probability ensemble for existing classification experiments."""

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from shared import (  # noqa: E402
    CLASSIFICATION_OUTPUT_DIR,
    DATA_DIR,
    EXPERIMENTS,
    LABELS,
    NUM_WORKERS,
    ExperimentConfig,
    ensure_experiment_dirs,
    get_experiment_config,
    load_module,
)

_here = Path(__file__).parent
_ds = load_module("dataset", _here / "02_dataset.py")
_mdl = load_module("model", _here / "03_model.py")
_predict = load_module("predict", _here / "06_predict.py")

VOCDataset = _ds.VOCDataset
load_train_df = _ds.load_train_df
load_test_df = _ds.load_test_df
get_val_transform = _ds.get_val_transform
MultiLabelClassifier = _mdl.MultiLabelClassifier
predict_tta = _predict.predict_tta
rle_encode = _predict.rle_encode

DEFAULT_ENSEMBLE_EXPERIMENTS = [
    "resnet50_224",
    "efficientnet_b3_320",
    "convnext_tiny_320",
    "convnext_small_320",
]
WEAK_CLASSES = ["bottle", "diningtable", "pottedplant", "sheep", "sofa"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["val-search", "predict"],
        required=True,
        help="val-search fits ensemble weights/thresholds; predict writes test submission.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=DEFAULT_ENSEMBLE_EXPERIMENTS,
        choices=sorted(EXPERIMENTS),
        help="Experiments to ensemble. Missing checkpoints are skipped by default.",
    )
    parser.add_argument(
        "--ckpt",
        choices=["best", "final", "last", "auto"],
        default="best",
        help="Checkpoint to use for each experiment. auto prefers final, then best.",
    )
    parser.add_argument(
        "--name",
        default="ensemble_selected",
        help="Output folder under output/image_classification/.",
    )
    parser.add_argument(
        "--weights-json",
        type=Path,
        default=None,
        help="Weights JSON from val-search. Defaults to <name>/metrics/ensemble_weights.json.",
    )
    parser.add_argument(
        "--grid-step",
        type=float,
        default=0.25,
        help="Weight grid step for global search.",
    )
    parser.add_argument(
        "--val-limit",
        type=int,
        default=None,
        help="Optional first-N validation rows for smoke tests.",
    )
    parser.add_argument(
        "--no-skip-missing",
        action="store_true",
        help="Fail instead of skipping experiments with missing checkpoints.",
    )
    parser.add_argument(
        "--no-tta",
        action="store_true",
        help="Disable horizontal-flip TTA when collecting probabilities.",
    )
    return parser.parse_args()


def output_paths(name: str) -> dict[str, Path]:
    root = CLASSIFICATION_OUTPUT_DIR / name
    paths = {
        "root": root,
        "metrics": root / "metrics",
        "predictions": root / "predictions",
        "submissions": root / "submissions",
        "figures": root / "figures",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def resolve_checkpoint(config: ExperimentConfig, ckpt: str) -> Path:
    if ckpt == "auto":
        return config.final_checkpoint if config.final_checkpoint.exists() else config.best_checkpoint
    if ckpt == "final":
        return config.final_checkpoint
    if ckpt == "last":
        return config.last_checkpoint
    return config.best_checkpoint


def available_configs(
    experiment_names: list[str],
    ckpt: str,
    skip_missing: bool,
) -> list[tuple[ExperimentConfig, Path]]:
    configs = []
    for name in experiment_names:
        config = get_experiment_config(name)
        ensure_experiment_dirs(config)
        checkpoint = resolve_checkpoint(config, ckpt)
        if checkpoint.exists():
            configs.append((config, checkpoint))
            continue
        message = f"Checkpoint not found for {name}: {checkpoint}"
        if skip_missing:
            print(f"[skip] {message}")
        else:
            raise FileNotFoundError(message)
    if not configs:
        raise FileNotFoundError("No usable experiment checkpoints found.")
    return configs


def build_val_df(config: ExperimentConfig, limit: int | None = None) -> pd.DataFrame:
    df = load_train_df(DATA_DIR)
    _, val_idx = train_test_split(
        range(len(df)),
        test_size=config.val_split,
        random_state=config.random_seed,
    )
    val_df = df.iloc[val_idx]
    if limit is not None:
        val_df = val_df.iloc[:limit]
    return val_df


def build_loader(df: pd.DataFrame, config: ExperimentConfig, split: str, device: torch.device):
    ds = VOCDataset(
        df,
        DATA_DIR,
        split=split,
        transform=get_val_transform(config.img_size, mode=config.transform_mode),
    )
    loader = DataLoader(
        ds,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    return ds, loader


def collect_probs(
    config: ExperimentConfig,
    checkpoint: Path,
    df: pd.DataFrame,
    split: str,
    device: torch.device,
    use_tta: bool,
) -> np.ndarray:
    model = MultiLabelClassifier(
        backbone=config.backbone,
        num_classes=len(LABELS),
        pretrained=False,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    _, loader = build_loader(df, config, split, device)
    all_probs = []
    desc = f"{config.name} ({split})"
    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc=desc, leave=False):
            imgs = imgs.to(device)
            if use_tta:
                probs = predict_tta(model, imgs)
            else:
                probs = torch.sigmoid(model(imgs))
            all_probs.append(probs.cpu().numpy())
    return np.vstack(all_probs)


def average_precision(probs: np.ndarray, labels: np.ndarray) -> tuple[pd.DataFrame, float]:
    rows = []
    for i, label in enumerate(LABELS):
        rows.append({"class": label, "ap": average_precision_score(labels[:, i], probs[:, i])})
    ap_df = pd.DataFrame(rows)
    return ap_df, float(ap_df["ap"].mean())


def find_best_thresholds(
    probs: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray = np.arange(0.1, 0.91, 0.05),
) -> np.ndarray:
    best_thresholds = []
    for class_idx in range(probs.shape[1]):
        best_t, best_f1 = 0.5, -1.0
        for threshold in thresholds:
            preds = (probs[:, class_idx] > threshold).astype(int)
            f1 = f1_score(labels[:, class_idx], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, threshold
        best_thresholds.append(best_t)
    return np.array(best_thresholds)


def threshold_report(
    probs: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray,
    prefix: str,
) -> pd.DataFrame:
    rows = []
    preds = (probs > thresholds[None, :]).astype(int)
    for i, label in enumerate(LABELS):
        y_true = labels[:, i].astype(int)
        y_pred = preds[:, i].astype(int)
        rows.append(
            {
                "class": label,
                f"{prefix}_f1": f1_score(y_true, y_pred, zero_division=0),
                f"{prefix}_tp": int(((y_pred == 1) & (y_true == 1)).sum()),
                f"{prefix}_fp": int(((y_pred == 1) & (y_true == 0)).sum()),
                f"{prefix}_fn": int(((y_pred == 0) & (y_true == 1)).sum()),
                f"{prefix}_threshold": thresholds[i],
            }
        )
    return pd.DataFrame(rows)


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    total = weights.sum()
    if total <= 0:
        raise ValueError("At least one ensemble weight must be positive.")
    return weights / total


def iter_weight_grid(num_models: int, step: float):
    values = np.arange(0.0, 1.0 + step / 2, step)
    for weights in product(values, repeat=num_models):
        weights = np.array(weights, dtype=float)
        if weights.sum() <= 0:
            continue
        yield normalize_weights(weights)


def weighted_average(prob_stack: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.tensordot(weights, prob_stack, axes=(0, 0))


def search_global_weights(prob_stack: np.ndarray, labels: np.ndarray, step: float):
    best = None
    for weights in iter_weight_grid(prob_stack.shape[0], step):
        probs = weighted_average(prob_stack, weights)
        _, map_score = average_precision(probs, labels)
        if best is None or map_score > best["mAP"]:
            best = {"weights": weights, "mAP": map_score}
    return best


def search_per_class_weights(prob_stack: np.ndarray, labels: np.ndarray, step: float):
    num_models = prob_stack.shape[0]
    class_weights = np.zeros((len(LABELS), num_models), dtype=float)
    class_ap = []
    candidate_weights = list(iter_weight_grid(num_models, step))
    for class_idx in range(len(LABELS)):
        best_ap, best_weights = -1.0, None
        for weights in candidate_weights:
            probs = np.tensordot(weights, prob_stack[:, :, class_idx], axes=(0, 0))
            ap = average_precision_score(labels[:, class_idx], probs)
            if ap > best_ap:
                best_ap, best_weights = ap, weights
        class_weights[class_idx] = best_weights
        class_ap.append(best_ap)
    return class_weights, float(np.mean(class_ap)), np.array(class_ap)


def apply_per_class_weights(prob_stack: np.ndarray, class_weights: np.ndarray) -> np.ndarray:
    output = np.zeros(prob_stack.shape[1:], dtype=float)
    for class_idx in range(len(LABELS)):
        output[:, class_idx] = np.tensordot(
            class_weights[class_idx],
            prob_stack[:, :, class_idx],
            axes=(0, 0),
        )
    return output


def choose_report(
    global_probs: np.ndarray,
    per_class_probs: np.ndarray,
    labels: np.ndarray,
) -> tuple[str, np.ndarray, np.ndarray, pd.DataFrame]:
    global_thresholds = find_best_thresholds(global_probs, labels)
    per_class_thresholds = find_best_thresholds(per_class_probs, labels)
    global_f1 = threshold_report(global_probs, labels, global_thresholds, "global")
    per_class_f1 = threshold_report(per_class_probs, labels, per_class_thresholds, "per_class")
    global_mean = float(global_f1["global_f1"].mean())
    per_class_mean = float(per_class_f1["per_class_f1"].mean())
    if per_class_mean >= global_mean:
        return "per_class", per_class_probs, per_class_thresholds, per_class_f1
    return "global", global_probs, global_thresholds, global_f1


def val_search(args) -> None:
    paths = output_paths(args.name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = available_configs(args.experiments, args.ckpt, skip_missing=not args.no_skip_missing)
    val_df = build_val_df(configs[0][0], limit=args.val_limit)
    labels = val_df[LABELS].to_numpy(dtype=float)

    prob_arrays = []
    for config, checkpoint in configs:
        prob_arrays.append(
            collect_probs(
                config,
                checkpoint,
                val_df,
                split="train",
                device=device,
                use_tta=not args.no_tta,
            )
        )
    prob_stack = np.stack(prob_arrays, axis=0)

    baseline_rows = []
    for (config, _), probs in zip(configs, prob_arrays):
        _, map_score = average_precision(probs, labels)
        thresholds = find_best_thresholds(probs, labels)
        f1_df = threshold_report(probs, labels, thresholds, config.name)
        weak_mean = float(f1_df[f1_df["class"].isin(WEAK_CLASSES)][f"{config.name}_f1"].mean())
        baseline_rows.append({"experiment": config.name, "mAP": map_score, "weak_class_f1": weak_mean})
    baseline_df = pd.DataFrame(baseline_rows)
    baseline_df.to_csv(paths["metrics"] / "ensemble_baselines.csv", index=False)

    global_best = search_global_weights(prob_stack, labels, args.grid_step)
    global_probs = weighted_average(prob_stack, global_best["weights"])
    class_weights, per_class_map, per_class_ap = search_per_class_weights(
        prob_stack,
        labels,
        args.grid_step,
    )
    per_class_probs = apply_per_class_weights(prob_stack, class_weights)
    chosen_mode, chosen_probs, thresholds, f1_df = choose_report(global_probs, per_class_probs, labels)

    ap_df, chosen_map = average_precision(chosen_probs, labels)
    ap_df["best_threshold"] = thresholds
    ap_df["per_class_weighted_ap"] = per_class_ap
    for i, (config, _) in enumerate(configs):
        ap_df[f"weight_{config.name}"] = class_weights[:, i]
    ap_df.to_csv(paths["metrics"] / "ap_per_class.csv", index=False)
    f1_df.to_csv(paths["metrics"] / "threshold_report.csv", index=False)
    pd.DataFrame(chosen_probs, columns=LABELS, index=val_df.index).to_csv(
        paths["predictions"] / "val_probabilities.csv",
        index_label="Id",
    )
    np.save(paths["metrics"] / "best_thresholds.npy", thresholds)

    weights_payload = {
        "name": args.name,
        "experiments": [config.name for config, _ in configs],
        "checkpoint": args.ckpt,
        "use_tta": not args.no_tta,
        "grid_step": args.grid_step,
        "chosen_mode": chosen_mode,
        "global_weights": global_best["weights"].tolist(),
        "global_mAP": global_best["mAP"],
        "per_class_mAP": per_class_map,
        "chosen_mAP": chosen_map,
        "thresholds": thresholds.tolist(),
        "per_class_weights": {
            label: class_weights[i].tolist() for i, label in enumerate(LABELS)
        },
    }
    weights_json = paths["metrics"] / "ensemble_weights.json"
    weights_json.write_text(json.dumps(weights_payload, indent=2), encoding="utf-8")

    summary = pd.DataFrame(
        [
            {
                "name": args.name,
                "experiments": ",".join(weights_payload["experiments"]),
                "chosen_mode": chosen_mode,
                "mAP": chosen_map,
                "weak_class_f1": float(
                    f1_df[f1_df["class"].isin(WEAK_CLASSES)].iloc[:, 1].mean()
                ),
                "val_rows": len(val_df),
            }
        ]
    )
    summary.to_csv(paths["metrics"] / "evaluation_summary.csv", index=False)

    print(f"Ensemble mode: {chosen_mode}")
    print(f"Included experiments: {', '.join(weights_payload['experiments'])}")
    print(f"Global mAP: {global_best['mAP']:.4f}")
    print(f"Per-class weighted mAP: {per_class_map:.4f}")
    print(f"Chosen mAP: {chosen_map:.4f}")
    print(f"Weights JSON: {weights_json}")
    print(f"Thresholds: {paths['metrics'] / 'best_thresholds.npy'}")


def load_weights(args) -> dict:
    weights_json = args.weights_json
    if weights_json is None:
        weights_json = output_paths(args.name)["metrics"] / "ensemble_weights.json"
    if not weights_json.exists():
        raise FileNotFoundError(f"Weights JSON not found: {weights_json}")
    return json.loads(weights_json.read_text(encoding="utf-8"))


def predict(args) -> None:
    payload = load_weights(args)
    paths = output_paths(payload.get("name", args.name))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = available_configs(payload["experiments"], payload["checkpoint"], skip_missing=False)
    test_df = load_test_df(DATA_DIR)

    prob_arrays = []
    for config, checkpoint in configs:
        prob_arrays.append(
            collect_probs(
                config,
                checkpoint,
                test_df,
                split="test",
                device=device,
                use_tta=payload.get("use_tta", True),
            )
        )
    prob_stack = np.stack(prob_arrays, axis=0)

    if payload["chosen_mode"] == "per_class":
        class_weights = np.array([payload["per_class_weights"][label] for label in LABELS])
        all_probs = apply_per_class_weights(prob_stack, class_weights)
    else:
        all_probs = weighted_average(prob_stack, np.array(payload["global_weights"]))

    thresholds = np.array(payload["thresholds"])
    preds = (all_probs > thresholds[None, :]).astype(int)
    all_indices = list(test_df.index)

    prob_df = pd.DataFrame(all_probs, columns=LABELS, index=all_indices)
    pred_df = pd.DataFrame(preds, columns=LABELS, index=all_indices)
    prob_df.index.name = "Id"
    pred_df.index.name = "Id"
    prob_path = paths["predictions"] / f"test_probabilities_{payload['name']}.csv"
    binary_path = paths["predictions"] / f"test_binary_predictions_{payload['name']}.csv"
    submission_path = paths["submissions"] / f"submission_classification_{payload['name']}.csv"
    prob_df.to_csv(prob_path)
    pred_df.to_csv(binary_path)

    rows = {"Id": [], "Predicted": []}
    for idx in pred_df.index:
        label_vec = pred_df.loc[idx, LABELS].values.astype(int)
        rows["Id"].append(f"{idx}_classification")
        rows["Predicted"].append(rle_encode(label_vec))
        rows["Id"].append(f"{idx}_segmentation")
        rows["Predicted"].append("")
    pd.DataFrame(rows).set_index("Id").to_csv(submission_path)

    print(f"Probabilities saved to {prob_path}")
    print(f"Binary predictions saved to {binary_path}")
    print(f"Submission saved to {submission_path}")


def main():
    args = parse_args()
    if args.mode == "val-search":
        val_search(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
