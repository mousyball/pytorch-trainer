from .base_hook import HOOKS, Hook


@HOOKS.register_module()
class OptimizerHook(Hook):

    def __init__(self, interval=1):
        """step torch optimizer from trainer every n iterations.

        Args:
            interval (int): the period to average loss
                and step optimizer.
        """
        self.interval = interval
        self.avg_grad_cnt = 0

    def after_train_iter(self, trainer):
        # TODO: how to call sum of multi-loss
        loss = trainer.outputs['loss']

        loss / self.interval
        loss.backward()
        self.avg_grad_cnt += 1

        if self.avg_grad_cnt % self.interval == 0:
            trainer.optimizer.step()
            trainer.optimizer.zero_grad()
            self.avg_grad_cnt = 0
