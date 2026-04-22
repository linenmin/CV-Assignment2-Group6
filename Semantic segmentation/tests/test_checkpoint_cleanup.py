from pathlib import Path

from ga2_seg.checkpoint_cleanup import prune_checkpoints, resolve_last_checkpoint_path


def test_resolve_last_checkpoint_path_reads_relative_path(tmp_path: Path):
    work_dir = tmp_path / "work_dir"
    work_dir.mkdir()
    checkpoint_path = work_dir / "iter_10.pth"
    checkpoint_path.write_bytes(b"checkpoint")
    (work_dir / "last_checkpoint").write_text("iter_10.pth\n", encoding="utf-8")

    assert resolve_last_checkpoint_path(work_dir) == checkpoint_path.resolve()


def test_prune_checkpoints_keeps_best_and_last_only(tmp_path: Path):
    work_dir = tmp_path / "work_dir"
    work_dir.mkdir()

    best_path = work_dir / "best_mIoU_iter_8.pth"
    last_path = work_dir / "iter_14.pth"
    delete_path = work_dir / "iter_8.pth"

    best_path.write_bytes(b"b" * 10)
    last_path.write_bytes(b"l" * 20)
    delete_path.write_bytes(b"d" * 30)
    (work_dir / "last_checkpoint").write_text("iter_14.pth\n", encoding="utf-8")

    result = prune_checkpoints(work_dir)

    assert best_path.exists()
    assert last_path.exists()
    assert not delete_path.exists()
    assert result["freed_bytes"] == 30
    assert len(result["deleted"]) == 1


def test_prune_checkpoints_can_delete_all_when_keep_policy_is_disabled(tmp_path: Path):
    work_dir = tmp_path / "work_dir"
    work_dir.mkdir()

    first_path = work_dir / "iter_1.pth"
    second_path = work_dir / "iter_2.pth"
    first_path.write_bytes(b"a")
    second_path.write_bytes(b"b")

    result = prune_checkpoints(work_dir, keep_best=False, keep_last=False)

    assert not first_path.exists()
    assert not second_path.exists()
    assert len(result["deleted"]) == 2
