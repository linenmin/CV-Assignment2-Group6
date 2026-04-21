import argparse
from pathlib import Path

from ga2_seg.paths import get_project_root
from ga2_seg.visualize import save_sample_grid, save_transform_preview


def generate_visualization_outputs(output_dir: Path, num_samples: int = 3) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_indices = list(range(num_samples))
    outputs = [
        save_sample_grid(output_dir / "sample_grid.png", sample_indices=sample_indices),
        save_transform_preview(output_dir / "transform_preview.png", sample_index=sample_indices[0]),
    ]
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize semantic segmentation samples.")
    parser.add_argument("--output-dir", type=Path, default=get_project_root() / "outputs" / "figures")
    parser.add_argument("--num-samples", type=int, default=3)
    args = parser.parse_args()
    outputs = generate_visualization_outputs(output_dir=args.output_dir, num_samples=args.num_samples)
    print(f"project_root={get_project_root()}")
    for output_path in outputs:
        print(f"generated={output_path}")


if __name__ == "__main__":
    main()
