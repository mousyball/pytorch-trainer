"""
File modify from https://github.com/open-mmlab/mmcv
License File Available at:
https://github.com/open-mmlab/mmcv/blob/master/LICENSE
"""
import time
import warnings

import torch

from .utils import get_host_info
from .base_trainer import TRAINER, BaseTrainer


@TRAINER.register_module()
class EpochBasedRunner(BaseTrainer):
    """Epoch-based Trainer."""

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.outputs = self.model.train_step(
                data_batch, self.optimizer, **kwargs)
            self.call_hook('after_train_iter')

        self.call_hook('after_train_epoch')
        self._epoch += 1

    @torch.no_grad()
    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_val_iter')
            self.run_iter(data_batch, train_mode=False)
            self.outputs = self.model.val_step(
                data_batch, self.optimizer, **kwargs)
            self.call_hook('after_val_iter')

        self.call_hook('after_val_epoch')

    def fit(self, data_loaders, workflow):
        """Start training.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                training order and epochs. E.g, [('train', 2), ('val', 1)]
                means training 2 epochs for training and 1 epoch for
                validation, iteratively.
        """
        # TODO: data_loader / workflow format tbd

        self.logger.info('Start running, host: {0}, work_dir: {1}'.format(
            get_host_info(), self.work_dir))
        self.logger.info(
            'workflow: {0}, max: {1:4d} epochs', workflow, self._max_epochs)

        self.call_hook('before_run')
        while self.epoch < self.max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                epoch_runner = getattr(self, mode)

                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= self.max_epochs:
                        break
                    epoch_runner(data_loaders[i])

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')


@TRAINER.register_module()
class Runner(EpochBasedRunner):
    """Deprecated name of EpochBasedRunner."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            'Runner was deprecated, please use EpochBasedRunner instead')
        super().__init__(*args, **kwargs)
