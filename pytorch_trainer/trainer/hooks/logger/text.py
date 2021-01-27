from ..base_hook import HOOKS
from .base_logger import LoggerHook


@HOOKS.register()
class TextLoggerHook(LoggerHook):
    def __init__(self, interval=500):
        """show current loss every n iterations.

        Args:
            interval (int): the iteration period to show current loss
        """
        self.interval = interval
        self.interval_cnt = 0

    def after_train_iter(self, trainer):
        self.interval_cnt += 1
        if self.interval_cnt % self.interval != 0:
            return

        # TODO: how to call sum of multi-loss
        loss_dict = trainer.outputs['multi_loss']
        for key, val in loss_dict.items():
            trainer.logger.info('epoch-{0} | iter-{1} | {2:8s}:{3:5f}'.format(
                trainer.epoch + 1, trainer.inner_iter + 1, key, val))
        self.interval_cnt = 0
