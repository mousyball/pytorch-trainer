import os.path as osp

from torch.utils.tensorboard import SummaryWriter

from .base_logger import LoggerHook

# TODO: registry
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
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def after_train_epoch(self, trainer):
        """log loss and lr group every n training epoch"""
        if not self.is_n_epoch(trainer, self.interval):
            return
        
        step = trainer.epoch + 1
        loss_dict = self.get_loss_log(trainer)
        for key, val in loss_dict.items():
            self.writer.add_scalar('Train/{0}_epoch'.format(key), val, step)
        
        lr_dict = self.get_lr_log(trainer)
        for key, val in lr_dict.items():
            self.writer.add_scalar('LR/{0}_epoch'.format(key), val, step)

    def after_train_batch(self, trainer):
        """log loss and lr group every n training batch iteration"""
        if not self.is_n_batch(trainer, self.interval):
            return

        step = trainer.batch_iter + 1
        loss_dict = self.get_loss_log(trainer)
        for key, val in loss_dict.items():
            self.writer.add_scalar('Train/{0}_batch_iter'.format(key), val, step)
        
        lr_dict = self.get_lr_log(trainer)
        for key, val in lr_dict.items():
            self.writer.add_scalar('LR/{0}_batch_iter'.format(key), val, step)
    
    def after_val_epoch(self, trainer):
        """log loss every evaluation epoch"""
        if trainer.epoch is None:
            step = trainer.batch_iter
        else:
            step = trainer.epoch

        loss_dict = self.get_loss_log(trainer)
        for key, val in loss_dict.items():
            self.writer.add_scalar('Val/{0}_epoch'.format(key), val, step)

    def after_run(self, trainer):
        self.writer.close()
