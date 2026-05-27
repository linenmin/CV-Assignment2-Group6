"""Embed the redesigned SVG diagrams directly inline in the matching
markdown cells of group6_final.ipynb, and drop the previous
matplotlib-imshow code cells (which produced a low-resolution raster
preview of the same diagrams).

Why inline SVG rather than a referenced file:
 - Jupyter and Kaggle both render <svg> tags inside markdown cells.
 - The diagram is vector, so it stays sharp at any zoom or DPI.
 - The notebook becomes self-contained: no runtime dependency on the
   bundled image-loading code cell or on a mermaid extension.

The script is idempotent: re-running it leaves a notebook in the same
state regardless of whether the previous diagrams are already present.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(
    r"D:\BaiduNetdiskWorkspace\Leuven\8th\Computer Vision\assignment\Group2\kaggle_submission"
)
NB_PATH = ROOT / "notebook" / "group6_final.ipynb"
SVG_DIR = ROOT / "dataset" / "figures"

SVG_BEGIN = "<svg "
SVG_END = "</svg>"


def load_svg(name: str) -> str:
    text = (SVG_DIR / name).read_text(encoding="utf-8").strip()
    if not text.startswith("<svg") or not text.endswith("</svg>"):
        raise SystemExit(f"Expected an SVG file: {name}")
    # Markdown treats lines indented by 4+ spaces as a code block, which
    # would dump the SVG source instead of rendering it. Strip leading
    # whitespace from every line and drop blank lines so the entire SVG
    # is parsed as inline HTML.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "".join(lines)


def strip_existing_svg(src: str) -> str:
    """Remove any inline <svg ...>...</svg> block from a markdown source."""
    out = src
    while True:
        start = out.find(SVG_BEGIN)
        if start == -1:
            break
        end = out.find(SVG_END, start)
        if end == -1:
            break
        out = out[:start].rstrip() + "\n\n" + out[end + len(SVG_END):].lstrip()
    return out.rstrip() + "\n"


def upsert_svg_in_markdown(cell: dict, svg_text: str, caption_text: str) -> bool:
    """Replace any existing SVG block + figure caption in the cell with
    the new SVG and caption. Returns True if changes were made."""
    src = "".join(cell.get("source", []))
    src_clean = strip_existing_svg(src)
    # Drop any pre-existing figure-caption paragraph. A caption is a line
    # that begins with `*Figure` and continues until the next blank line.
    lines = src_clean.split("\n")
    keep_lines = []
    drop = False
    for line in lines:
        if not drop and line.lstrip().startswith("*Figure"):
            drop = True
            continue
        if drop:
            if line.strip() == "":
                drop = False
            continue
        keep_lines.append(line)
    src_clean = "\n".join(keep_lines).rstrip() + "\n"

    new_src = src_clean + "\n" + svg_text + "\n\n" + caption_text + "\n"
    if new_src == src:
        return False
    cell["source"] = new_src.splitlines(keepends=True)
    return True


def remove_cell_by_id(nb: dict, cell_id: str) -> bool:
    cells = nb["cells"]
    for i, c in enumerate(cells):
        if c.get("id") == cell_id:
            del cells[i]
            return True
    return False


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

    ch2_svg = load_svg("ch2_convnext_architecture.svg")
    ch3_svg = load_svg("ch3_eomt_dataflow.svg")
    ch4_svg = load_svg("ch4_attack_modes.svg")

    ch2_caption = (
        "*Figure 0.* ConvNeXt-Small (Liu et al., 2022) on a `320 × 320` "
        "input. Blue blocks carry ImageNet-1K pretrained weights; "
        "the orange head is the only fully new component, trained "
        "from scratch on PASCAL VOC. The lower row zooms into one "
        "ConvNeXt block: depthwise `7 × 7` conv for spatial mixing, "
        "LayerNorm in place of BatchNorm so small batches stay stable, "
        "an inverted bottleneck (`1 × 1` expand 4×, GELU, `1 × 1` "
        "project) for channel mixing, then a residual add so gradients "
        "flow through all `36` blocks without vanishing."
    )

    ch3_caption = (
        "*Figure 1.* EoMT-DINOv3 (Kerssies et al., 2025) on a frozen DINOv3 "
        "ViT-L/16 backbone. Blue boxes carry pretrained DINOv3 weights and "
        "stay frozen during PASCAL VOC fine-tuning; orange boxes (the last "
        "two ViT layers plus 100 mask queries) are the only trainable "
        "parameters. Green boxes produce the per-query mask and class "
        "logits that are fused into one semantic map by argmax + bilinear "
        "upsample."
    )

    ch4_caption = (
        "*Figure 2.* Both pipelines hit the same frozen V17 EoMT-DINOv3 with "
        "the same `ε = 6 / 255` budget. **(A)** trains one generator `G(x; θ)` "
        "across the training distribution (the assignment PDF Figure 3 "
        "setup); the dashed orange path is the gradient flow that updates "
        "θ once per epoch. **(B)** keeps no generator: for every test image "
        "it initialises δ at zero and refines it with PGD inner-loop "
        "updates. Chapter 4.3 reports that A fails completely on V17 "
        "(mIoU drop ≈ 0) while B succeeds completely (pixel accuracy "
        "drops from 99 % to under 8 %)."
    )

    n_changed = 0
    for c in nb["cells"]:
        if c.get("id") == "ch2-convnext-components":
            if upsert_svg_in_markdown(c, ch2_svg, ch2_caption):
                n_changed += 1
        if c.get("id") == "ch3-eomt-components":
            if upsert_svg_in_markdown(c, ch3_svg, ch3_caption):
                n_changed += 1
        if c.get("id") == "ch4-attack-modes-diagram":
            if upsert_svg_in_markdown(c, ch4_svg, ch4_caption):
                n_changed += 1

    n_removed = 0
    for cell_id in ("ch3-eomt-image", "ch4-attack-modes-image"):
        if remove_cell_by_id(nb, cell_id):
            n_removed += 1

    NB_PATH.write_text(
        json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Markdown cells with embedded SVG updated: {n_changed}")
    print(f"Old matplotlib-imshow code cells removed: {n_removed}")
    print(f"Notebook total cells: {len(nb['cells'])}")


if __name__ == "__main__":
    main()
