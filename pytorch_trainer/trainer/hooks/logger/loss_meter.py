from ..base_hook import HOOKS
from .base_logger import LoggerHook


@HOOKS.register()
class LossLoggerHook(LoggerHook):
    """
    loss_meters belongs to trainer rather than LossLoggerHook()'.
    To make loss_meters independent between Hook
    LossLoggerHook only update and clear trainer loss_meters
    """

    def after_train_iter(self, trainer):
        loss_dict = trainer.outputs['multi_loss']
        trainer.loss_meters.update(loss_dict)

    def before_val_epoch(self, trainer):
        trainer.loss_meters.clear()

    def before_val_batch(self, trainer):
        trainer.loss_meters.clear()

    def after_val_iter(self, trainer):
        loss_dict = trainer.outputs['multi_loss']
        trainer.loss_meters.update(loss_dict)
