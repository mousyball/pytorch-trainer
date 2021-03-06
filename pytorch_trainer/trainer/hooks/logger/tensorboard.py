import os
import os.path as osp

from torch.utils.tensorboard import SummaryWriter

from ..base_hook import HOOKS
from .base_logger import LoggerHook


@HOOKS.register()
class TensorboardLoggerHook(LoggerHook):
    def __init__(self,
                 interval=1,
                 log_dir=None,
                 ):
        """Log data to tensorboard periodically.

        Args:
            interval (int): the logging peroid. if save at after_train_epoch
                interval indicates epochs, otherwise it indicates batch iterations.
            log_dir (str): tensorboard logging path
        """
        super(TensorboardLoggerHook, self).__init__(interval)
        self.log_dir = log_dir
        self.interval = interval

    def before_run(self, trainer):
        # TODO: version check. SummaryWriter only support pytorch version >= 1.14

        if self.log_dir is None:
            self.log_dir = osp.join(trainer.work_dir, 'tensorboard')
        if not osp.isdir(self.log_dir):
            os.makedirs(self.log_dir)
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def after_train_epoch(self, trainer):
        """log loss and lr group every n training epoch"""
        if not self.is_n_epoch(trainer, self.interval):
            return

        step = trainer.epoch + 1
        self._add_loss_scalar(trainer, step)
        self._add_lr_scalar(trainer, step)

    def after_train_batch(self, trainer):
        """log loss and lr group every n training batch iteration"""
        if not self.is_n_batch(trainer, self.interval):
            return

        step = trainer.iter + 1
        self._add_loss_scalar(trainer, step)
        self._add_lr_scalar(trainer, step)

    def after_val_epoch(self, trainer):
        """log loss every evaluation epoch"""
        step = trainer.epoch + 1
        self._add_loss_scalar(trainer, step)

    def after_val_batch(self, trainer):
        """log loss every evaluation iteration"""
        step = trainer.iter + 1
        self._add_loss_scalar(trainer, step)

    def after_run(self, trainer):
        self.writer.close()

    def _add_loss_scalar(self, trainer, step):
        trainer_mode = self.get_trainer_mode(trainer).capitalize()
        trainer_base = self.get_trainer_base(trainer)
        loss_dict = self.get_loss_log(trainer)
        for key, val in loss_dict.items():
            self.writer.add_scalar(
                '{0}/{1}_{2}'.format(trainer_mode, key, trainer_base), val, step)

    def _add_lr_scalar(self, trainer, step):
        trainer_base = self.get_trainer_base(trainer)
        lr_dict = self.get_lr_log(trainer)
        for key, val in lr_dict.items():
            self.writer.add_scalar(
                'LR/{0}_{1}'.format(key, trainer_base), val, step)
