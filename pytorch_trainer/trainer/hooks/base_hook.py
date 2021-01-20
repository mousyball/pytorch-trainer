from pytorch_trainer.utils import Registry

HOOKS = Registry('hook')

class Hook:

    def before_run(self):
        pass

    def after_run(self):
        pass

    def before_train_epoch(self):
        """abstract method for epoch_based_trainer only"""
        pass

    def after_train_epoch(self):
        """abstract method for epoch_based_trainer only"""
        pass

    def before_train_batch(self):
        """abstract method for batch_iter_based_trainer only"""
        pass

    def after_train_batch(self):
        """abstract method for batch_iter_based_trainer only"""
        pass

    def before_train_iter(self):
        pass

    def after_train_iter(self):
        pass

    # TODO: valdation size
    def before_val_epoch(self):
        pass

    def before_val_iter(self):
        pass

    def after_val_iter(self):
        pass

    def after_val_epoch(self):
        pass

    def is_n_epoch(self, trainer, n):
        return (trainer.epoch + 1) % n == 0 if n > 0 else False

    def is_n_batches(self, trainer, n):
        # TODO:
        raise NotImplementedError
