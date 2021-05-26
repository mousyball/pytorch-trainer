from .base_hook import HOOKS, Hook


@HOOKS.register()
class ModelHook(Hook):

    def before_run(self, trainer):
        trainer.model.to(trainer.device)
