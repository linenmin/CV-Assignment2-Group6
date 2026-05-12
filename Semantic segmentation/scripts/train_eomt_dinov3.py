"""Fine-tune EoMT-DINOv3 on the GA2 semantic segmentation split."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, EomtDinov3ForUniversalSegmentation

from ga2_seg.labels import ALL_CLASSES, NUM_CLASSES_WITH_BACKGROUND


MODEL_ID = "tue-mps/eomt-dinov3-ade-semantic-large-512"
IGNORE_INDEX = 255


@dataclass(frozen=True)
class Sample:
    stem: str
    image_path: Path
    mask_path: Path | None


class GA2SegmentationDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        split: str,
        image_size: int,
        training: bool,
    ) -> None:
        self.data_root = data_root
        self.split = split
        self.image_size = image_size
        self.training = training
        self.samples = self._load_samples()

    def _load_samples(self) -> list[Sample]:
        split_file = self.data_root / f"{self.split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        if self.split == "test":
            image_dir = self.data_root / "images" / "test"
            mask_dir = None
        elif self.split == "validation":
            image_dir = self.data_root / "images" / "validation"
            mask_dir = self.data_root / "annotations" / "validation"
        elif self.split == "training":
            image_dir = self.data_root / "images" / "training"
            mask_dir = self.data_root / "annotations" / "training"
        else:
            raise ValueError(f"Unsupported split: {self.split}")

        samples: list[Sample] = []
        for line in split_file.read_text(encoding="utf-8").splitlines():
            stem = line.strip()
            if not stem:
                continue
            image_path = image_dir / f"{stem}.png"
            mask_path = None if mask_dir is None else mask_dir / f"{stem}.png"
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            if mask_path is not None and not mask_path.exists():
                raise FileNotFoundError(f"Mask not found: {mask_path}")
            samples.append(Sample(stem=stem, image_path=image_path, mask_path=mask_path))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        original_size = image.size[::-1]
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)

        mask_array = None
        if sample.mask_path is not None:
            mask = Image.open(sample.mask_path)
            mask = mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
            mask_array = np.asarray(mask, dtype=np.int64)

        if self.training and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if mask_array is not None:
                mask_array = np.ascontiguousarray(mask_array[:, ::-1])

        return {
            "stem": sample.stem,
            "image": image,
            "mask": mask_array,
            "original_size": original_size,
        }


def semantic_mask_to_instances(mask: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    class_ids = [int(v) for v in np.unique(mask) if int(v) != IGNORE_INDEX]
    if not class_ids:
        raise ValueError("Mask has no valid classes.")
    masks = np.stack([(mask == class_id).astype(np.float32) for class_id in class_ids], axis=0)
    return torch.from_numpy(masks), torch.tensor(class_ids, dtype=torch.long)


def collate_train(batch: list[dict[str, object]], processor: AutoImageProcessor) -> dict[str, object]:
    images = [item["image"] for item in batch]
    inputs = processor(images=images, return_tensors="pt")
    mask_labels: list[torch.Tensor] = []
    class_labels: list[torch.Tensor] = []
    for item in batch:
        mask = item["mask"]
        if mask is None:
            raise ValueError("Training batch contains a sample without mask.")
        masks, classes = semantic_mask_to_instances(mask)
        mask_labels.append(masks)
        class_labels.append(classes)
    inputs["mask_labels"] = mask_labels
    inputs["class_labels"] = class_labels
    return inputs


def collate_eval(batch: list[dict[str, object]], processor: AutoImageProcessor) -> dict[str, object]:
    images = [item["image"] for item in batch]
    inputs = processor(images=images, return_tensors="pt")
    inputs["masks"] = [item["mask"] for item in batch]
    inputs["stems"] = [item["stem"] for item in batch]
    inputs["original_sizes"] = [item["original_size"] for item in batch]
    return inputs


def move_batch_to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    moved: dict[str, object] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        elif key in {"mask_labels", "class_labels"}:
            moved[key] = [tensor.to(device) for tensor in value]  # type: ignore[arg-type]
        else:
            moved[key] = value
    return moved


def build_model(model_id: str, device: torch.device) -> EomtDinov3ForUniversalSegmentation:
    id2label = {index: label for index, label in enumerate(ALL_CLASSES)}
    label2id = {label: index for index, label in id2label.items()}
    model = EomtDinov3ForUniversalSegmentation.from_pretrained(
        model_id,
        num_labels=NUM_CLASSES_WITH_BACKGROUND,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
        dtype="auto",
    )
    return model.to(device)


def set_trainable_scope(model: torch.nn.Module, unfreeze_last_layers: int) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False

    for module_name in ("class_predictor", "mask_head"):
        module = getattr(model, module_name)
        for parameter in module.parameters():
            parameter.requires_grad = True

    if unfreeze_last_layers > 0:
        encoder = getattr(model, "eomt", None) or getattr(model, "dinov3", None) or model
        layers = None
        for attr in ("encoder", "layer", "layers"):
            candidate = getattr(encoder, attr, None) if encoder is not None else None
            if isinstance(candidate, torch.nn.ModuleList):
                layers = candidate
                break
        if layers is not None:
            for layer in layers[-unfreeze_last_layers:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True


def build_param_groups(
    model: torch.nn.Module,
    base_lr: float,
    weight_decay: float,
    unfreeze_last_layers: int,
    backbone_lr: float | None,
    backbone_lr_decay: float,
) -> list[dict[str, object]]:
    if backbone_lr is None:
        return [
            {
                "params": [parameter for parameter in model.parameters() if parameter.requires_grad],
                "lr": base_lr,
                "weight_decay": weight_decay,
                "name": "trainable",
            }
        ]

    head_params: list[torch.nn.Parameter] = []
    for module_name in ("class_predictor", "mask_head"):
        module = getattr(model, module_name)
        head_params.extend([parameter for parameter in module.parameters() if parameter.requires_grad])

    param_groups: list[dict[str, object]] = [
        {
            "params": head_params,
            "lr": base_lr,
            "weight_decay": weight_decay,
            "name": "heads",
        }
    ]

    layers = getattr(model, "layers", None)
    if isinstance(layers, torch.nn.ModuleList) and unfreeze_last_layers > 0:
        selected_layers = list(layers[-unfreeze_last_layers:])
        for offset, layer in enumerate(selected_layers):
            distance_from_output = unfreeze_last_layers - offset - 1
            layer_lr = backbone_lr * (backbone_lr_decay ** distance_from_output)
            layer_params = [parameter for parameter in layer.parameters() if parameter.requires_grad]
            if layer_params:
                param_groups.append(
                    {
                        "params": layer_params,
                        "lr": layer_lr,
                        "weight_decay": weight_decay,
                        "name": f"backbone_layer_{len(layers) - unfreeze_last_layers + offset}",
                    }
                )

    return param_groups


def compute_lr_scale(step: int, warmup_steps: int, max_steps: int, min_lr_ratio: float) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return max((step + 1) / warmup_steps, min_lr_ratio)
    if max_steps <= warmup_steps:
        return 1.0
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def compute_miou(pred: np.ndarray, target: np.ndarray, num_classes: int) -> tuple[float, dict[int, float]]:
    per_class: dict[int, float] = {}
    for class_id in range(num_classes):
        valid = target != IGNORE_INDEX
        pred_mask = (pred == class_id) & valid
        target_mask = (target == class_id) & valid
        union = np.logical_or(pred_mask, target_mask).sum()
        if union == 0:
            continue
        intersection = np.logical_and(pred_mask, target_mask).sum()
        per_class[class_id] = float(intersection / union)
    return float(np.mean(list(per_class.values()))) if per_class else 0.0, per_class


def evaluate(
    model: EomtDinov3ForUniversalSegmentation,
    processor: AutoImageProcessor,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[float, dict[int, float]]:
    model.eval()
    intersections = np.zeros(NUM_CLASSES_WITH_BACKGROUND, dtype=np.float64)
    unions = np.zeros(NUM_CLASSES_WITH_BACKGROUND, dtype=np.float64)

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = move_batch_to_device(batch, device)
            outputs = model(pixel_values=batch["pixel_values"])
            predictions = processor.post_process_semantic_segmentation(
                outputs,
                target_sizes=[(512, 512)] * len(batch["masks"]),
            )
            for pred_tensor, target in zip(predictions, batch["masks"]):
                pred = pred_tensor.detach().cpu().numpy().astype(np.int64)
                target_array = np.asarray(target, dtype=np.int64)
                for class_id in range(NUM_CLASSES_WITH_BACKGROUND):
                    valid = target_array != IGNORE_INDEX
                    pred_mask = (pred == class_id) & valid
                    target_mask = (target_array == class_id) & valid
                    intersections[class_id] += np.logical_and(pred_mask, target_mask).sum()
                    unions[class_id] += np.logical_or(pred_mask, target_mask).sum()

    per_class = {
        class_id: float(intersections[class_id] / unions[class_id])
        for class_id in range(NUM_CLASSES_WITH_BACKGROUND)
        if unions[class_id] > 0
    }
    miou = float(np.mean(list(per_class.values()))) if per_class else 0.0
    return miou, per_class


def save_checkpoint(
    output_dir: Path,
    name: str,
    model: EomtDinov3ForUniversalSegmentation,
    processor: AutoImageProcessor,
    metrics: dict[str, float],
) -> Path:
    checkpoint_dir = output_dir / name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    processor.save_pretrained(checkpoint_dir)
    (checkpoint_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return checkpoint_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/segman_ga2")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-dir", default="outputs/checkpoints/eomt_dinov3_v11")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--early-stop-patience", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=None)
    parser.add_argument("--backbone-lr-decay", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--unfreeze-last-layers", type=int, default=0)
    parser.add_argument("--scheduler", choices=("constant", "cosine"), default="constant")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke-steps", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoImageProcessor.from_pretrained(args.model_id)
    model = build_model(args.model_id, device)
    set_trainable_scope(model, args.unfreeze_last_layers)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable_params={trainable}", flush=True)
    print(f"total_params={total}", flush=True)

    train_dataset = GA2SegmentationDataset(Path(args.data_root), "training", args.image_size, training=True)
    val_dataset = GA2SegmentationDataset(Path(args.data_root), "validation", args.image_size, training=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: collate_train(batch, processor),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: collate_eval(batch, processor),
    )

    param_groups = build_param_groups(
        model=model,
        base_lr=args.lr,
        weight_decay=args.weight_decay,
        unfreeze_last_layers=args.unfreeze_last_layers,
        backbone_lr=args.backbone_lr,
        backbone_lr_decay=args.backbone_lr_decay,
    )
    base_lrs = [float(group["lr"]) for group in param_groups]
    optimizer = torch.optim.AdamW(param_groups)
    print(
        "param_groups="
        + json.dumps(
            [
                {
                    "name": group.get("name", f"group_{index}"),
                    "lr": float(group["lr"]),
                    "params": sum(parameter.numel() for parameter in group["params"]),
                }
                for index, group in enumerate(param_groups)
            ],
            indent=2,
        ),
        flush=True,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    max_steps = args.smoke_steps if args.smoke_steps > 0 else args.max_steps

    log_path = output_dir / "training_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "loss", "val_miou", "lr", "elapsed_sec"])
        writer.writeheader()

    best_miou = -1.0
    best_step = 0
    no_improve = 0
    step = 0
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)

    while step < max_steps:
        model.train()
        for batch in train_loader:
            if args.scheduler == "cosine":
                lr_scale = compute_lr_scale(
                    step=step,
                    warmup_steps=args.warmup_steps,
                    max_steps=max_steps,
                    min_lr_ratio=args.min_lr_ratio,
                )
                for group, base_lr in zip(optimizer.param_groups, base_lrs):
                    group["lr"] = base_lr * lr_scale

            batch = move_batch_to_device(batch, device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                outputs = model(
                    pixel_values=batch["pixel_values"],
                    mask_labels=batch["mask_labels"],
                    class_labels=batch["class_labels"],
                )
                loss = outputs.loss / args.grad_accum

            scaler.scale(loss).backward()
            if (step + 1) % args.grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            step += 1
            raw_loss = float(loss.detach().cpu().item() * args.grad_accum)
            should_eval = step == 1 or step % args.eval_every == 0 or step >= max_steps
            if should_eval:
                eval_batches = 4 if args.smoke_steps > 0 else None
                val_miou, per_class = evaluate(model, processor, val_loader, device, max_batches=eval_batches)
                elapsed = time.perf_counter() - start
                row = {
                    "step": step,
                    "loss": raw_loss,
                    "val_miou": val_miou,
                    "lr": optimizer.param_groups[0]["lr"],
                    "elapsed_sec": elapsed,
                }
                with log_path.open("a", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=row.keys())
                    writer.writerow(row)
                print(
                    f"step={step} loss={raw_loss:.4f} val_miou={val_miou:.4f} elapsed_sec={elapsed:.1f}",
                    flush=True,
                )
                (output_dir / "latest_metrics.json").write_text(
                    json.dumps(
                        {
                            **row,
                            "best_miou": best_miou,
                            "best_step": best_step,
                            "per_class_iou": per_class,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                save_checkpoint(output_dir, "last", model, processor, {"step": step, "val_miou": val_miou})
                if val_miou > best_miou:
                    best_miou = val_miou
                    best_step = step
                    no_improve = 0
                    save_checkpoint(output_dir, "best", model, processor, {"step": step, "val_miou": val_miou})
                else:
                    no_improve += 1

                if args.smoke_steps <= 0 and no_improve >= args.early_stop_patience:
                    print(f"early_stop step={step} best_step={best_step} best_miou={best_miou:.4f}", flush=True)
                    return

            if step >= max_steps:
                break

    print(f"finished step={step} best_step={best_step} best_miou={best_miou:.4f}", flush=True)


if __name__ == "__main__":
    main()
