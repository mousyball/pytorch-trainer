import copy
import shutil
import os.path as osp
import tempfile
from glob import glob

import torch
import pytest
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from pytorch_trainer.utils import get_cfg_defaults
from pytorch_trainer.trainer import IterBasedTrainer, EpochBasedTrainer


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)

    def train_step(self, x):
        return dict(loss=self(x))

    def val_step(self, x):
        return dict(loss=self(x))


class Test_epoch_based:
    config = get_cfg_defaults()
    config.merge_from_file('configs/pytorch_trainer/hook/trainer_hook.yaml')

    @pytest.mark.parametrize("cfg, expected",
                             [
                                 ({}, -1.65),
                                 ({'interval': 2}, -0.43)
                             ])
    def test_optimzier_hook(self, cfg, expected):
        # config
        config = copy.deepcopy(self.config)
        config.merge_from_list(['LOGGER_HOOK.NAME', []])
        config.merge_from_list(['HOOK.NAME', ['OptimizerHook']])
        config.HOOK.OptimizerHook.update(cfg)

        # model
        torch.manual_seed(0)
        model = Model()
        data_loader = DataLoader(torch.ones((5, 2)))
        trainer = EpochBasedTrainer(model,
                                    optimizer=optim.SGD(
                                        model.parameters(), lr=0.02),
                                    max_epoch=5)
        trainer.register_callback(config)
        trainer.fit([data_loader], [('train', 1)])
        assert round(trainer.outputs['loss'].item(), 2) == expected

    @pytest.mark.parametrize("cfg, expected",
                             [
                                 (dict(mode='default'), 0.004),
                                 ((dict(interval=2, mode='step')), 0.0008)
                             ])
    def test_scheduler_hook(self, cfg, expected):
        # config
        config = copy.deepcopy(self.config)
        config.merge_from_list(['LOGGER_HOOK.NAME', []])
        config.merge_from_list(
            ['HOOK.NAME', ['OptimizerHook', 'SchedulerHook']])
        config.HOOK.SchedulerHook.update(cfg)

        # model
        torch.manual_seed(0)
        model = Model()
        data_loader = DataLoader(torch.ones((5, 2)))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
        scheduler = StepLR(optimizer, step_size=2, gamma=0.2)
        trainer = EpochBasedTrainer(model,
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    max_epoch=2)
        trainer.register_callback(config)
        trainer.fit([data_loader], [('train', 1)])
        assert trainer.optimizer.param_groups[0]['lr'] == expected

    @pytest.mark.parametrize("cfg, expected",
                             [
                                 (dict(interval=2), 'epoch_002.pth')
                             ])
    def test_checkpoint_hook(self, cfg, expected):
        # config
        config = copy.deepcopy(self.config)
        config.merge_from_list(['LOGGER_HOOK.NAME', []])
        config.merge_from_list(
            ['HOOK.NAME', ['OptimizerHook', 'CheckpointHook']])
        config.HOOK.CheckpointHook.update(cfg)

        # temporarily work directory
        work_dir = tempfile.mkdtemp()

        # model
        model = Model()
        data_loader = DataLoader(torch.ones((5, 2)))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
        trainer = EpochBasedTrainer(model,
                                    optimizer=optimizer,
                                    work_dir=work_dir,
                                    max_epoch=2)
        trainer.register_callback(config)
        trainer.fit([data_loader], [('train', 1)])
        output_path = osp.join(trainer.work_dir, 'checkpoint', '*.pth')

        assert osp.basename(glob(output_path)[0]) == expected

        shutil.rmtree(work_dir)
        shutil.rmtree(trainer.work_dir)


class Test_iter_based:
    config = get_cfg_defaults()
    config.merge_from_file('configs/pytorch_trainer/hook/trainer_hook.yaml')

    @pytest.mark.parametrize("cfg, expected",
                             [
                                 ({}, -0.75),
                                 ({'interval': 2}, -0.16)
                             ])
    def test_optimzier_hook(self, cfg, expected):
        # config
        config = copy.deepcopy(self.config)
        config.merge_from_list(['LOGGER_HOOK.NAME', []])
        config.merge_from_list(['HOOK.NAME', ['OptimizerHook']])
        config.HOOK.OptimizerHook.update(cfg)

        # model
        torch.manual_seed(0)
        model = Model()
        data_loader = DataLoader(torch.ones((5, 2)))

        trainer = IterBasedTrainer(model,
                                   optimizer=optim.SGD(
                                       model.parameters(), lr=0.02),
                                   max_iter=10)
        trainer.register_callback(config)
        trainer.fit([data_loader], [('train', 1)])
        assert round(trainer.outputs['loss'].item(), 2) == expected

    @pytest.mark.parametrize("cfg, expected",
                             [
                                 (dict(mode='default'), 0.004),
                                 ((dict(interval=1, mode='step')), 0.0008)
                             ])
    def test_scheduler_hook(self, cfg, expected):
        # config
        config = copy.deepcopy(self.config)
        config.merge_from_list(['LOGGER_HOOK.NAME', []])
        config.merge_from_list(
            ['HOOK.NAME', ['OptimizerHook', 'SchedulerHook']])
        config.HOOK.SchedulerHook.update(cfg)

        torch.manual_seed(0)
        model = Model()
        data_loader = DataLoader(torch.ones((5, 2)))

        optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
        scheduler = StepLR(optimizer, step_size=2, gamma=0.2)

        trainer = IterBasedTrainer(model,
                                   optimizer=optimizer,
                                   scheduler=scheduler,
                                   max_iter=4)
        trainer.register_callback(config)
        trainer.fit([data_loader], [('train', 2)])

        assert trainer.optimizer.param_groups[0]['lr'] == expected

    @pytest.mark.parametrize("cfg, expected",
                             [
                                 (dict(interval=2), 'iter_0000002.pth')
                             ])
    def test_checkpoint_hook(self, cfg, expected):
        # config
        config = copy.deepcopy(self.config)
        config.merge_from_list(['LOGGER_HOOK.NAME', []])
        config.merge_from_list(
            ['HOOK.NAME', ['OptimizerHook', 'CheckpointHook']])
        config.HOOK.CheckpointHook.update(cfg)

        # temporarily work directory
        work_dir = tempfile.mkdtemp()

        # model
        model = Model()
        data_loader = DataLoader(torch.ones((5, 2)))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
        trainer = IterBasedTrainer(model,
                                   optimizer=optimizer,
                                   work_dir=work_dir,
                                   max_iter=2)
        trainer.register_callback(config)
        trainer.fit([data_loader], [('train', 1)])
        output_path = osp.join(trainer.work_dir, 'checkpoint', '*.pth')

        assert osp.basename(glob(output_path)[0]) == expected

        shutil.rmtree(work_dir)
        shutil.rmtree(trainer.work_dir)

    # TODO: logger hook test
