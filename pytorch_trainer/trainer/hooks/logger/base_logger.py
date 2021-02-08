from ..base_hook import Hook


class LoggerHook(Hook):
    """Base class for logger hooks.
    Args:
        interval (int): Logging interval (every k iterations).
    """

    def __init__(self,
                 interval=1,
                 ):
        self.interval = interval

    def get_trainer_mode(self, trainer):
        return trainer.mode

    def get_trainer_base(self, trainer):
        return trainer.base

    def get_lr_log(self, trainer):
        """Return a training lr dict {'param_group_n':float}"""
        # TODO: assign name for each params group?
        return {f'param_group_{ii}': val['lr'] for ii, val in enumerate(trainer.optimizer.param_groups)}

    def get_loss_log(self, trainer):
        """get train or val loss meter and clear from trainer"""
        loss_meter = trainer.loss_meters[self.__class__.__name__]
        loss_dict = {key: val.avg for key, val in loss_meter.meters.items()}
        loss_meter.clear_meter()
        return loss_dict
