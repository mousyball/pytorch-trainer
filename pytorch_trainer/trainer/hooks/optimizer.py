from .base_hook import Hook, HOOKS


@HOOKS.register_module()
class OptimizerHook(Hook):

    def __init__(self, average_grad_period=1):
        """step torch optimizer from trainer every n iterations.

        Args:
            average_grad_period (int): the period to average loss
            and step optimizer.
        """
        self.avg_grad_period = average_grad_period
        self.avg_grad_cnt = 0

    def after_train_iter(self, trainer):
        # TODO: how to call sum of multi-loss
        loss = trainer.outputs['loss']

        loss / self.avg_grad_period
        loss.backward()
        self.avg_grad_cnt += 1

        if self.avg_grad_cnt % self.avg_grad_period == 0:
            trainer.optimizer.step()
            trainer.optimizer.zero_grad()
            self.avg_grad_cnt = 0
