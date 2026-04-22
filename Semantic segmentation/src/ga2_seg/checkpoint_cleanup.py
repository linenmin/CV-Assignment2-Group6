from __future__ import annotations

from pathlib import Path


def resolve_last_checkpoint_path(work_dir: str | Path) -> Path | None:
    work_dir = Path(work_dir).resolve()
    last_checkpoint_file = work_dir / "last_checkpoint"
    if not last_checkpoint_file.exists():
        return None

    checkpoint_path = last_checkpoint_file.read_text(encoding="utf-8").strip()
    if not checkpoint_path:
        return None

    path = Path(checkpoint_path)
    if not path.is_absolute():
        path = work_dir / path
    return path.resolve()


def prune_checkpoints(
    work_dir: str | Path,
    *,
    keep_best: bool = True,
    keep_last: bool = True,
) -> dict[str, object]:
    """Delete redundant checkpoints in one training work_dir.

    By default this keeps all `best_*.pth` checkpoints plus the checkpoint
    pointed to by `last_checkpoint`, and deletes the remaining `.pth` files
    directly under the given `work_dir`.
    """
    work_dir = Path(work_dir).resolve()
    checkpoint_paths = sorted(work_dir.glob("*.pth"))

    keep_paths: set[Path] = set()
    if keep_best:
        keep_paths.update(path.resolve() for path in work_dir.glob("best_*.pth"))
    if keep_last:
        last_path = resolve_last_checkpoint_path(work_dir)
        if last_path is not None:
            keep_paths.add(last_path)

    deleted_paths: list[Path] = []
    freed_bytes = 0
    for checkpoint_path in checkpoint_paths:
        resolved_path = checkpoint_path.resolve()
        if resolved_path in keep_paths:
            continue
        freed_bytes += checkpoint_path.stat().st_size
        checkpoint_path.unlink()
        deleted_paths.append(resolved_path)

    return {
        "work_dir": work_dir,
        "kept": sorted(str(path) for path in keep_paths if path.exists()),
        "deleted": sorted(str(path) for path in deleted_paths),
        "freed_bytes": freed_bytes,
    }
