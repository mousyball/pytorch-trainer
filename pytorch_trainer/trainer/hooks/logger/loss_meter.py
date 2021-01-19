from .base_logger import LoggerHook

class LossLoggerHook(LoggerHook):
    """
    meter belongs to trainer rather than LossLoggerHook()'.
    To make meter independent between Hook
    LossLoggerHook only update and clear trainer meter
    """

    def before_train_epoch(self, trainer):
        trainer.meter.clear()
    
    def before_train_batch(self, trainer):
        trainer.meter.clear()

    def after_train_iter(self, trainer):
        # TODO: define loss dict
        loss_dict = trainer.outputs['multi_loss']
        trainer.meter.update(loss_dict)

    def before_val_epoch(self, trainer):
        trainer.meter.clear()

    def after_val_iter(self, trainer):
        # TODO: define loss dict
        loss_dict = trainer.outputs['multi_loss']
        trainer.meter.update()
