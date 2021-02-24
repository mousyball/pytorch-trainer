import time

import torch

from .utils import IterDataLoader, sync_counter, get_host_info
from .base_trainer import TRAINER, BaseTrainer


@TRAINER.register()
class IterBasedTrainer(BaseTrainer):
    """Iter-based trainer"""

    def __init__(self,
                 model,
                 max_iter=0,
                 optimizer=None,
                 scheduler=None,
                 work_dir=None,
                 logger=None,
                 meta=None):
        super().__init__(model, max_iter=max_iter, optimizer=optimizer,
                         scheduler=scheduler, work_dir=work_dir, logger=logger, meta=meta)
        self.base = 'iter'
        self._max_inner_iter = 0

    @property
    def max_inner_iter(self):
        return self._max_inner_iter

    def make_iterator(self, data_loaders):
        return [IterDataLoader(data_loader) for data_loader in data_loaders]

    def train(self, data_loader, **kwargs):
        self.mode = 'train'
        self.model.train()
        self.call_hook('before_train_batch')
        time.sleep(2)
        # make iteration based behavior similar to epoch based trainer
        for i in range(self.max_inner_iter):
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.outputs = self.model.train_step(next(data_loader), **kwargs)
            self.outputs = self._loss_parser(self.outputs)
            self.call_hook('after_train_iter')
            self._iter += 1

            if self.iter >= self.max_iter:
                break

        self.call_hook_with_sync('after_train_batch')

    @torch.no_grad()
    @sync_counter
    def val(self, data_loader, **kwargs):
        self.mode = 'val'
        self.model.eval()
        self.call_hook('before_val_batch')
        time.sleep(2)
        for i in range(self.max_inner_iter):
            self._inner_iter = i
            self.call_hook('before_val_iter')
            self.outputs = self.model.val_step(next(data_loader))
            self.outputs = self._loss_parser(self.outputs)
            self.call_hook('after_val_iter')

        self.call_hook('after_val_batch')

    def fit(self, data_loaders, workflow):
        """Start training

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
        self.logger.info('workflow: {0}, max: {1:4d} iterations'.format(
            workflow, self.max_iter))

        data_loaders = self.make_iterator(data_loaders)

        self.call_hook('before_run')
        while self.iter < self.max_iter:
            for i, flow in enumerate(workflow):
                mode, iterations = flow

                # assign maximum inner iteration
                if iterations == -1:
                    self._max_inner_iter = len(data_loaders[i])
                else:
                    self._max_inner_iter = iterations

                iter_trainer = getattr(self, mode)
                iter_trainer(data_loaders[i])
