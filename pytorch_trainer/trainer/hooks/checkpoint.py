import os.path as osp

import torch

from .base_hook import Hook, HOOKS


@HOOKS.register_module()
class CheckpointHook(Hook):

    def __init__(self,
                 interval=-1,
                 out_dir=None,
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
        self.save_optimizer = save_optimizer

    def _save_checkpoint(self, trainer, by_epoch):
        if self.out_dir is None:
            self.out_dir = trainer.work_dir

        save = dict(model=trainer.model.state_dict(),
                    optim=trainer.optimizer.state_dict())

        if by_epoch:
            save.update(dict(epoch=trainer.epoch))
            ckpt_path = osp.join(
                self.out_dir, 'epoch_{}.pth'.format(trainer.epoch))
        else:
            save.update(dict(batch_iter=trainer.batch_iter))
            ckpt_path = osp.join(
                self.out_dir, 'batch_iter_{}.pth'.format(trainer.batch_iter))

        torch.save(save, ckpt_path)

    def after_train_epoch(self, trainer):
        if not self.is_n_epochs(trainer, self.interval):
            return

        trainer.logger.info('saving epoch {0}'.format(trainer.epoch + 1))
        self._save_checkpoint(trainer, by_epoch=True)

    def after_train_batch(self, trainer):
        if not self.is_n_batches(trainer, self.interval):
            return

        trainer.logger.info(
            'saving batch iter {0}'.format(trainer.batch_iter + 1))
        self._save_checkpoint(trainer, by_epoch=False)
