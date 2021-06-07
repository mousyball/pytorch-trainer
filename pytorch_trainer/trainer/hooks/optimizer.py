from .base_hook import HOOKS, Hook


@HOOKS.register()
class OptimizerHook(Hook):

    def __init__(self, interval=1):
        """step torch optimizer from trainer every n iterations.

        Args:
            interval (int): the period to average loss
                and step optimizer.
        """
        self.interval = interval
        self.avg_grad_cnt = 0

    def _get_trainer_loss(self, trainer):
        """return trainer loss summation"""
        return trainer.outputs['loss']

    def before_train_epoch(self, trainer):
        trainer.optimizer.zero_grad()

    def before_train_batch(self, trainer):
        trainer.optimizer.zero_grad()

    def after_train_iter(self, trainer):
        loss = self._get_trainer_loss(trainer)

        loss /= self.interval
        loss.backward()
        self.avg_grad_cnt += 1

        if self.avg_grad_cnt % self.interval == 0:
            trainer.optimizer.step()
            trainer.optimizer.zero_grad()
            self.avg_grad_cnt = 0
