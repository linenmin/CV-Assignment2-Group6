import argparse
import json
from pathlib import Path

from ga2_seg.runtime_compat import configure_windows_runtime

configure_windows_runtime()

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from ga2_seg.paths import get_dataset_root, get_project_root
from ga2_seg.stats import build_stats_summary


def _plot_bar(data: dict[str, int], output_path: Path, title: str, xlabel: str, ylabel: str) -> None:
    labels = list(data.keys())
    values = list(data.values())
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(range(len(labels)), values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=70, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def generate_analysis_outputs(output_dir: Path, metadata_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    stats = build_stats_summary()
    stats_path = metadata_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    _plot_bar(
        stats["class_presence_counts"],
        output_dir / "class_presence_counts.png",
        title="Train Set Class Presence Counts",
        xlabel="Class",
        ylabel="Image Count",
    )
    _plot_bar(
        stats["pixel_label_counts"],
        output_dir / "pixel_label_counts.png",
        title="Train Set Pixel Label Counts",
        xlabel="Label Id",
        ylabel="Pixel Count",
    )
    image_shape_counts = stats["image_shape_counts"]
    top_shapes = dict(list(image_shape_counts.items())[:20])
    _plot_bar(
        top_shapes,
        output_dir / "image_shape_counts_top20.png",
        title="Top 20 Train Image Shapes",
        xlabel="Shape",
        ylabel="Count",
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the semantic segmentation dataset.")
    parser.add_argument("--output-dir", type=Path, default=get_project_root() / "outputs" / "figures")
    parser.add_argument("--metadata-dir", type=Path, default=get_project_root() / "data" / "metadata")
    args = parser.parse_args()

    stats = generate_analysis_outputs(output_dir=args.output_dir, metadata_dir=args.metadata_dir)
    print(f"project_root={get_project_root()}")
    print(f"dataset_root={get_dataset_root()}")
    print(f"train_examples={stats['train_examples']}")
    print(f"stats_json={args.metadata_dir / 'stats.json'}")


if __name__ == "__main__":
    main()
