from .base_hook import HOOKS, Hook


@HOOKS.register_module()
class SchedulerHook(Hook):
    def __init__(self,
                 interval='step',
                 average_grad_period=1):
        """step scheduler from trainer by specific interval

        Args:
            interval (str): the period of step scheduler. if interval = 'step'
                scheduler step every 'average_grad_period' iteration. otherwise
                step when calling 'after_train_epoch' or 'after_train_batch'
            average_grad_period (int, optional): [description]. Defaults to 1.
        """
        self.interval = interval
        self.avg_grad_period = average_grad_period
        self.avg_grad_cnt = 0

    def after_train_epoch(self, trainer):
        trainer.scheduler.step()

    def after_train_batch(self, trainer):
        trainer.scheduler.step()

    def after_train_iter(self, trainer):
        if self.interval != 'step':
            return

        # step every n batch iteration
        self.avg_grad_cnt += 1
        if self.avg_grad_cnt % self.avg_grad_period == 0:
            trainer.scheduler.step()
            self.avg_grad_cnt = 0
