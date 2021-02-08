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

    def get_epoch(self, trainer):
        return trainer.epoch + 1

    def get_iter(self, trainer):
        """Get the current training iteration step."""
        return trainer.iter + 1

    def get_inner_iter(self, trainer):
        """Get the current training iteration step within epochs.
           which means step <= dataloader length
        """
        return trainer.inner_iter + 1

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

    def get_all_logs(self,
                     trainer,
                     ignored_key=None):
        """return base log information."""
        # TODO: return all logable information as dict(key=scalr) or dic(key=string)
        log_info = dict()
        log_info.update(self.get_lr(trainer))
        log_info.update(self.get_loss(trainer))

        return log_info
