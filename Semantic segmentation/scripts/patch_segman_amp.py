from __future__ import annotations

from pathlib import Path


PATCH_MARKER = "# GA2 selective_scan AMP compatibility patch"


def main() -> None:
    seg_root = Path(__file__).resolve().parents[1] / "external" / "SegMAN"
    target = (
        seg_root
        / "segmentation"
        / "mmseg"
        / "models"
        / "decode_heads"
        / "segman_decoder.py"
    )
    text = target.read_text(encoding="utf-8")
    if PATCH_MARKER in text:
        print(f"already_patched={target}")
        return

    old = "        token_mix_feat = self.token_mixer(self.norm1(x))\n"
    new = (
        f"        {PATCH_MARKER}\n"
        "        token_mix_feat = self.token_mixer(\n"
        "            self.norm1(x), to_dtype=True, force_fp32=True)\n"
    )
    if old not in text:
        raise RuntimeError(f"Expected SegMAN token mixer call was not found in {target}")

    target.write_text(text.replace(old, new), encoding="utf-8")
    print(f"patched={target}")


if __name__ == "__main__":
    main()
