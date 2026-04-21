from mmengine.hooks import EarlyStoppingHook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class DelayedEarlyStoppingHook(EarlyStoppingHook):
    """Delay early-stopping checks until a minimum number of validations."""

    def __init__(self, begin: int = 0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.begin = begin
        self._num_records = 0

    def after_val_epoch(self, runner, metrics) -> None:
        self._num_records += 1
        if self.monitor not in metrics:
            return super().after_val_epoch(runner, metrics)

        current_score = metrics[self.monitor]
        if self._num_records <= self.begin:
            compare = self.rule_map[self.rule]
            if compare(self.best_score + self.min_delta, current_score):
                return

            self.best_score = current_score
            self.wait_count = 0
            return

        super().after_val_epoch(runner, metrics)
