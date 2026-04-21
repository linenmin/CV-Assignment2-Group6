from types import SimpleNamespace

from ga2_seg.early_stopping import DelayedEarlyStoppingHook


class _DummyLogger:
    def __init__(self) -> None:
        self.messages = []

    def info(self, message: str) -> None:
        self.messages.append(message)


def _build_runner():
    return SimpleNamespace(
        train_loop=SimpleNamespace(stop_training=False),
        logger=_DummyLogger(),
    )


def test_delayed_early_stopping_waits_until_begin_records():
    hook = DelayedEarlyStoppingHook(
        monitor="mIoU",
        rule="greater",
        min_delta=0.1,
        patience=2,
        begin=2,
    )
    runner = _build_runner()

    hook.before_run(runner)
    hook.after_val_epoch(runner, {"mIoU": 1.0})
    hook.after_val_epoch(runner, {"mIoU": 1.0})

    assert runner.train_loop.stop_training is False
    assert hook.best_score == 1.0
    assert hook.wait_count == 0


def test_delayed_early_stopping_stops_after_patience_once_active():
    hook = DelayedEarlyStoppingHook(
        monitor="mIoU",
        rule="greater",
        min_delta=0.1,
        patience=2,
        begin=1,
    )
    runner = _build_runner()

    hook.before_run(runner)
    hook.after_val_epoch(runner, {"mIoU": 1.0})
    hook.after_val_epoch(runner, {"mIoU": 1.05})
    hook.after_val_epoch(runner, {"mIoU": 1.08})

    assert runner.train_loop.stop_training is True
    assert hook.wait_count == 2
    assert runner.logger.messages
