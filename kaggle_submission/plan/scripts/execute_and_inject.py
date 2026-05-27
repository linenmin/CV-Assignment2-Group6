"""Manually execute the code cells of group6_final.ipynb and inject
their outputs back into the notebook JSON.

This bypasses the ipykernel startup issue in the local gpu_env. It
walks the notebook, runs every code cell's source in a shared exec
context, captures stdout and matplotlib figures, and writes the
notebook back with proper nbformat v4 outputs. The script is
idempotent: re-running it overwrites any previous outputs.
"""

from __future__ import annotations

import base64
import contextlib
import io
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NOTEBOOK_PATH = Path(__file__).resolve().parents[2] / "notebook" / "group6_final.ipynb"


def _capture_figures() -> list[str]:
    """Snapshot all open matplotlib figures as base64 PNGs, then close."""
    pngs: list[str] = []
    for num in plt.get_fignums():
        fig = plt.figure(num)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        pngs.append(base64.b64encode(buf.getvalue()).decode("ascii"))
        plt.close(fig)
    return pngs


def _capture_object(obj) -> dict | None:
    """Build an nbformat display_data dict for an object using its rich reprs."""
    data: dict[str, object] = {}
    if hasattr(obj, "_repr_html_"):
        html = obj._repr_html_()
        if html:
            data["text/html"] = html
    if hasattr(obj, "_repr_png_"):
        png = obj._repr_png_()
        if png:
            data["image/png"] = base64.b64encode(png).decode("ascii")
    if not data:
        try:
            text = repr(obj)
        except Exception:
            return None
        data["text/plain"] = text
    return {"output_type": "display_data", "data": data, "metadata": {}}


def main() -> None:
    import ast

    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    namespace: dict[str, object] = {"__name__": "__main__"}

    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell.get("source", []))
        outputs: list[dict] = []
        stdout_buf = io.StringIO()
        captured_displays: list = []

        # Install a capturing display() into the cell's namespace.
        def _fake_display(*objs, **_kwargs):
            for obj in objs:
                captured_displays.append(obj)

        namespace["display"] = _fake_display

        # Detect a final expression so we can capture its value (Jupyter
        # auto-displays the repr of the last expression).
        last_expr_var = None
        try:
            tree = ast.parse(source, mode="exec")
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last_expr_var = f"__cell_result__{id(cell)}"
                expr_node = tree.body[-1]
                assign_node = ast.Assign(
                    targets=[ast.Name(id=last_expr_var, ctx=ast.Store())],
                    value=expr_node.value,
                )
                ast.copy_location(assign_node, expr_node)
                tree.body[-1] = assign_node
                ast.fix_missing_locations(tree)
                code = compile(tree, f"<cell-{cell.get('id', '?')}>", "exec")
            else:
                code = compile(source, f"<cell-{cell.get('id', '?')}>", "exec")
        except SyntaxError:
            code = compile(source, f"<cell-{cell.get('id', '?')}>", "exec")
            last_expr_var = None

        try:
            with contextlib.redirect_stdout(stdout_buf):
                exec(code, namespace)
        except Exception as exc:  # surface failure but keep going
            tb_str = f"{type(exc).__name__}: {exc}\n"
            outputs.append(
                {
                    "output_type": "error",
                    "ename": type(exc).__name__,
                    "evalue": str(exc),
                    "traceback": [tb_str],
                }
            )

        stdout_text = stdout_buf.getvalue()
        if stdout_text:
            outputs.append(
                {
                    "output_type": "stream",
                    "name": "stdout",
                    "text": stdout_text,
                }
            )

        # Explicit display(...) calls first.
        for obj in captured_displays:
            captured = _capture_object(obj)
            if captured is not None:
                outputs.append(captured)

        # Then the last expression value, if any.
        if last_expr_var is not None and last_expr_var in namespace:
            value = namespace.pop(last_expr_var, None)
            if value is not None:
                captured = _capture_object(value)
                if captured is not None:
                    outputs.append(captured)

        # Matplotlib figures last.
        for png_b64 in _capture_figures():
            outputs.append(
                {
                    "output_type": "display_data",
                    "data": {"image/png": png_b64, "text/plain": ["<Figure>"]},
                    "metadata": {},
                }
            )

        cell["outputs"] = outputs
        cell["execution_count"] = 1

    NOTEBOOK_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Updated {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
