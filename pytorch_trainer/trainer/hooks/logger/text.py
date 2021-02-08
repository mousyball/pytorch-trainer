from ..base_hook import HOOKS
from .base_logger import LoggerHook


@HOOKS.register()
class TextLoggerHook(LoggerHook):
    def __init__(self, interval=500):
        """show current loss every n iterations.

        Args:
            interval (int): the iteration period to show current loss
        """
        self.mode = 'None'
        self.interval = interval
        self.interval_cnt = 0

    def after_train_iter(self, trainer):
        self.interval_cnt += 1
        if self.interval_cnt % self.interval != 0:
            return

        if self.get_trainer_base(trainer) == 'iter':
            self._iter_based_log(trainer)
        elif self.get_trainer_base(trainer) == 'epoch':
            self._epoch_based_log(trainer)

        self.interval_cnt = 0

    def _epoch_based_log(self, trainer):
        step = trainer.epoch + 1
        loss_dict = self.get_loss_log(trainer)
        for key, val in loss_dict.items():
            trainer.logger.info('epoch-{0} | iter-{1:6d} | {2:8s}:{3:5f}'.format(
                step, trainer.inner_iter + 1, key, val))

    def _iter_based_log(self, trainer):
        step = trainer.iter + 1
        loss_dict = self.get_loss_log(trainer)
        for key, val in loss_dict.items():
            trainer.logger.info('| iter-{0:8d} | {1:8s}:{2:5f}'.format(
                step, key, val))
