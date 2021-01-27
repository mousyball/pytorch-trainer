import os
import os.path as osp

import torch

from .base_hook import HOOKS, Hook


@HOOKS.register()
class CheckpointHook(Hook):

    def __init__(self,
                 interval=-1,
                 out_dir=None,
                 save_meta=False,
                 save_optimizer=True,
                 ):
        """Save checkpoints periodically.

        Args:
            interval (int): the saving peroid. if save at after_train_epoch
                interval indicates epochs, otherwise it indicates batch iterations.
            out_dir (str): The directory to save checkpoints
            save_optimizer (bool): Whether to save optimizer state_dict in the
            checkpoint
        """
        self.out_dir = out_dir
        self.interval = interval
        self.save_meta = save_meta
        self.save_optimizer = save_optimizer

    def _save_checkpoint(self, trainer, by_epoch):
        # saving path
        if self.out_dir is None:
            self.out_dir = osp.join(trainer.work_dir, './checkpoint/')
        if not osp.isdir(self.out_dir):
            os.makedirs(self.out_dir)

        # data need to save
        save = dict(model=trainer.model.state_dict())

        if self.save_optimizer:
            save.update(dict(optim=trainer.optimizer.state_dict()))

        if self.save_meta:
            # TODO: save meta data to checkpoint
            pass

        if by_epoch:
            save.update(dict(epoch=trainer.epoch + 1))
            ckpt_path = osp.join(
                self.out_dir, 'epoch_{}.pth'.format(str(trainer.epoch + 1).zfill(3)))
        else:
            save.update(dict(batch_iter=trainer.batch_iter + 1))
            ckpt_path = osp.join(
                self.out_dir, 'batch_iter_{}.pth'.format(str(trainer.batch_iter + 1).zfill(7)))

        torch.save(save, ckpt_path)

    def after_train_epoch(self, trainer):
        if not self.is_n_epoch(trainer, self.interval):
            return

        trainer.logger.info('saving epoch {0}'.format(trainer.epoch + 1))
        self._save_checkpoint(trainer, by_epoch=True)

    def after_train_batch(self, trainer):
        if not self.is_n_batches(trainer, self.interval):
            return

        trainer.logger.info(
            'saving batch iter {0}'.format(trainer.batch_iter + 1))
        self._save_checkpoint(trainer, by_epoch=False)
