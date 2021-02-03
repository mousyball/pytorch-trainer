from .base_hook import HOOKS, Hook


@HOOKS.register()
class SchedulerHook(Hook):
    def __init__(self,
                 mode='default',
                 interval=1):
        """step scheduler from trainer by specific interval

        Args:
            mode (str): If interval = 'step' scheduler step every 'interval' iteration.
                otherwise step when calling 'after_train_epoch' or 'after_train_batch'
            interval (int, optional): step scheduler every n interval. Defaults to 1.
        """
        self.mode = mode
        self.interval = interval
        self.avg_grad_cnt = 0

    def after_train_epoch(self, trainer):
        if self.mode == 'step':
            return

        trainer.scheduler.step()

    def after_train_batch(self, trainer):
        if self.mode == 'step':
            return

        trainer.scheduler.step()

    def after_train_iter(self, trainer):
        if self.mode != 'step':
            return

        # step every n batch iteration
        self.avg_grad_cnt += 1
        if self.avg_grad_cnt % self.interval == 0:
            trainer.scheduler.step()
            self.avg_grad_cnt = 0
