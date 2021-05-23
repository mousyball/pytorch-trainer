"""
File is modified from https://github.com/open-mmlab/mmcv
License File Available at:
https://github.com/open-mmlab/mmcv/blob/master/LICENSE
"""
import time

import torch

from .utils import sync_counter, get_host_info
from .profiling import profiling
from .base_trainer import TRAINER, BaseTrainer


@TRAINER.register()
class EpochBasedTrainer(BaseTrainer):
    """Epoch-based Trainer."""

    def __init__(self,
                 model,
                 max_epoch=0,
                 optimizer=None,
                 scheduler=None,
                 device=None,
                 work_dir=None,
                 logger=None,
                 meta=None):
        super().__init__(model, max_epoch=max_epoch, optimizer=optimizer,
                         scheduler=scheduler, device=device, work_dir=work_dir, logger=logger, meta=meta)
        self.base = 'epoch'

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_train_iter')
            data_batch = self.data_to_device(data_batch)
            self.outputs = self.model.train_step(data_batch, **kwargs)
            self.outputs = self._loss_parser(self.outputs)
            self.call_hook('after_train_iter')

        self.call_hook('after_train_epoch')
        self._epoch += 1

    @torch.no_grad()
    @sync_counter
    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_val_iter')
            data_batch = self.data_to_device(data_batch)
            self.outputs = self.model.val_step(data_batch, **kwargs)
            self.outputs = self._loss_parser(self.outputs)
            self.call_hook('after_val_iter')

        self.call_hook('after_val_epoch')

    @profiling
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
        self.logger.info('workflow: {0}, max: {1:4d} epochs'.format(
            workflow, self.max_epoch))

        self.call_hook('before_run')
        while self.epoch < self.max_epoch:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                epoch_trainer = getattr(self, mode)

                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= self.max_epoch:
                        break
                    epoch_trainer(data_loaders[i])

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')
