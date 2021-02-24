from ..base_hook import HOOKS
from .base_logger import LoggerHook


@HOOKS.register()
class LossLoggerHook(LoggerHook):
    """
    loss_meters belongs to trainer rather than 'LossLoggerHook()'.
    Trainer make loss_meters independent among hooks.
    'LossLoggerHook()' only update and clear trainer loss_meters.
    """

    def _get_loss(self, trainer):
        """get all current loss from trainer. Return Dictionary as
           dict(loss1=`torch.Tensor`, loss2=`torch.Tensor`). Tensor
           size is (1,1)
        """
        return trainer.outputs['multi_loss']

    def after_train_iter(self, trainer):
        loss_dict = self._get_loss(trainer)
        for loss_meter in trainer.loss_meters.values():
            loss_meter.update(loss_dict)

    def before_val_epoch(self, trainer):
        for loss_meter in trainer.loss_meters.values():
            loss_meter.clear_meter()

    def before_val_batch(self, trainer):
        for loss_meter in trainer.loss_meters.values():
            loss_meter.clear_meter()

    def after_val_iter(self, trainer):
        loss_dict = self._get_loss(trainer)
        for loss_meter in trainer.loss_meters.values():
            loss_meter.update(loss_dict)
