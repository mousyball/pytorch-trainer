import copy
import shutil
import tempfile

import torch
import pytest
import torch.nn as nn
from torch.utils.data import DataLoader

from pytorch_trainer.utils import get_cfg_defaults
from pytorch_trainer.trainer import IterBasedTrainer, EpochBasedTrainer
from pytorch_trainer.trainer.hooks import (
    OptimizerHook, SchedulerHook, CheckpointHook
)
from pytorch_trainer.trainer.base_trainer import BaseTrainer


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.linear.weight.data.fill_(0.5)
        self.linear.bias.data.fill_(0.1)

    def forward(self, x):
        return self.linear(x)

    def train_step(self, x):
        return dict(loss=self(x)[0, 0])
        # return dict(loss=self(x))

    def val_step(self, x):
        return dict(loss=self(x)[0, 0])
        # return dict(loss=self(x))


class model_wrapper:
    def __init__(self):
        self.module = Model()


class Test_base_trainer:

    @pytest.mark.xfail(reason='test raise error')
    def test_work_dir(self):
        BaseTrainer(Model(), work_dir=123)

    def test_load_parallel_model(self):
        model = model_wrapper()
        trainer = BaseTrainer(model)
        assert trainer._model_name == 'Model'


class Test_epoch_based_trainer:
    # config
    config = get_cfg_defaults()
    config.merge_from_file('configs/pytorch_trainer/hook/trainer_hook.yaml')
    config.merge_from_list(['HOOK.OptimizerHook.interval', 2])
    config.merge_from_list(['LOGGER_HOOK.NAME', []])

    # model
    model = Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
    trainer = EpochBasedTrainer(model,
                                optimizer=optimizer,
                                max_epoch=1)
    trainer.register_callback(config)
    data_loader = DataLoader(torch.ones((1, 2)))

    def test_val(self):
        self.trainer._epoch = 0
        self.trainer.fit([self.data_loader, self.data_loader], [
                         ('train', 1), ('val', 1)])

        assert round(self.trainer.outputs['loss'].item(), 2) == 1.1

    def test_flow(self):
        self.trainer._epoch = 0
        self.trainer.fit([self.data_loader], [('train', 2)])

        assert self.trainer.epoch == 1


class Test_iter_based_trainer:
    # config
    config = get_cfg_defaults()
    config.merge_from_file('configs/pytorch_trainer/hook/trainer_hook.yaml')
    config.merge_from_list(['HOOK.OptimizerHook.interval', 2])
    config.merge_from_list(['LOGGER_HOOK.NAME', []])

    # model
    model = Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
    trainer = IterBasedTrainer(model,
                               optimizer=optimizer,
                               max_iter=1)
    trainer.register_callback(config)
    data_loader = DataLoader(torch.ones((1, 2)))

    def test_val(self):
        self.trainer._iter = 0
        self.trainer.fit([self.data_loader, self.data_loader], [
                         ('train', 1), ('val', 1)])

        assert round(self.trainer.outputs['loss'].item(), 2) == 1.1

    def test_max_inner_iter(self):
        self.trainer._iter = 0
        self.trainer.fit([DataLoader(torch.ones((5, 2)))], [('train', -1)])

        assert self.trainer.max_inner_iter == 5

    def test_data_loader(self):
        self.trainer._iter = 0
        self.trainer._max_iter = 2
        data_loader = DataLoader(torch.arange(4).view(2, 2).type(torch.float))
        self.trainer.fit([data_loader], [('train', 1)])

        assert round(self.trainer.outputs['loss'].item(), 1) == 1.3


class Test_register_callback():
    config = get_cfg_defaults()
    config.merge_from_file('configs/pytorch_trainer/hook/trainer_hook.yaml')

    @pytest.mark.parametrize('cfg', [
        dict(),
        dict(interval=2),
    ])
    def test_regist_optimizer_hook(self, cfg):
        # config
        config = copy.deepcopy(self.config)
        config.merge_from_list(['HOOK.NAME', ['OptimizerHook']])
        config.HOOK.OptimizerHook.update(cfg)

        # trainer
        trainer = EpochBasedTrainer('model')
        trainer._register_hook(config.HOOK)
        for hook in trainer._hooks:
            if isinstance(hook, OptimizerHook):
                assert True and len(trainer._hooks) == 1
                return
            else:
                continue
        assert False

    @pytest.mark.parametrize('cfg', [
        dict(),
        dict(interval=2, mode='step'),
    ])
    def test_regist_scheduler_hook(self, cfg):
        # config
        config = copy.deepcopy(self.config)
        config.merge_from_list(['HOOK.NAME', ['SchedulerHook']])
        config.HOOK.SchedulerHook.update(cfg)

        # trainer
        trainer = EpochBasedTrainer('model',
                                    scheduler='not_none')

        trainer._register_hook(config.HOOK)
        for hook in trainer._hooks:
            if isinstance(hook, SchedulerHook):
                assert True and len(trainer._hooks) == 1
                return
            else:
                continue
        assert False

    @pytest.mark.parametrize('cfg', [
        dict(),
        dict(interval=2),
    ])
    def test_regist_checkpoint_hook(self, cfg):
        # config
        config = copy.deepcopy(self.config)
        config.merge_from_list(['HOOK.NAME', ['CheckpointHook']])
        config.HOOK.CheckpointHook.update(cfg)

        # trainer
        trainer = EpochBasedTrainer('model')

        trainer._register_hook(config.HOOK)
        for hook in trainer._hooks:
            if isinstance(hook, CheckpointHook):
                assert True and len(trainer._hooks) == 1
                return
            else:
                continue
        assert False

    def test_regist_all(self):
        config = copy.deepcopy(self.config)
        trainer = EpochBasedTrainer('model',
                                    scheduler='not_none')

        trainer.register_callback(config)

        assert len(trainer._hooks) == 6


class Test_text_logger:
    # config
    config = get_cfg_defaults()
    config.merge_from_file('configs/pytorch_trainer/hook/trainer_hook.yaml')
    config.merge_from_list(['HOOK.OptimizerHook.interval', 10])
    config.merge_from_list(
        ['LOGGER_HOOK.NAME', ['LossLoggerHook', 'TextLoggerHook']])
    config.merge_from_list(['LOGGER_HOOK.TextLoggerHook.interval', 3])

    # data loader
    data_loader = DataLoader(torch.arange(10).view(5, 2).type(torch.float))

    # model
    model = Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02)

    # trainer
    epoch_trainer = EpochBasedTrainer(model,
                                      optimizer=optimizer,
                                      max_epoch=1)
    epoch_trainer.register_callback(copy.deepcopy(config))

    # trainer
    iter_trainer = IterBasedTrainer(model,
                                    optimizer=optimizer,
                                    max_iter=5)
    iter_trainer.register_callback(copy.deepcopy(config))

    @pytest.mark.parametrize('mode', [
        'train',
        'val'
    ])
    def test_epoch_base(self, mode):
        self.epoch_trainer.fit([self.data_loader], [(mode, 1)])
        loss_meter = self.epoch_trainer.loss_meters['TextLoggerHook']

        assert round(loss_meter.meters['loss'].avg.item(), 2) == 7.6

    @pytest.mark.parametrize('mode', [
        'train',
        'val'
    ])
    def test_iter_base(self, mode):
        self.iter_trainer.fit([self.data_loader], [(mode, 5)])
        loss_meter = self.iter_trainer.loss_meters['TextLoggerHook']

        assert round(loss_meter.meters['loss'].avg.item(), 2) == 7.6


class Test_tensorboard_logger:
    # config
    config = get_cfg_defaults()
    config.merge_from_file('configs/pytorch_trainer/hook/trainer_hook.yaml')
    config.merge_from_list(['HOOK.OptimizerHook.interval', 100])
    config.merge_from_list(
        ['LOGGER_HOOK.NAME', ['LossLoggerHook', 'TensorboardLoggerHook']])
    config.merge_from_list(['LOGGER_HOOK.TensorboardLoggerHook.interval', 2])

    # data loader
    data_loader = DataLoader(torch.arange(10).view(5, 2).type(torch.float))

    # model
    model = Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02)

    work_dir = tempfile.mkdtemp()

    # trainer
    epoch_trainer = EpochBasedTrainer(model,
                                      work_dir=work_dir,
                                      optimizer=optimizer,
                                      max_epoch=3)
    epoch_trainer.register_callback(copy.deepcopy(config))

    # trainer
    iter_trainer = IterBasedTrainer(model,
                                    work_dir=work_dir,
                                    optimizer=optimizer,
                                    max_iter=5)
    iter_trainer.register_callback(copy.deepcopy(config))

    def test_epoch_base(self):
        self.epoch_trainer.fit([self.data_loader], [('train', 1)])
        loss_meter = self.epoch_trainer.loss_meters['TensorboardLoggerHook']

        assert round(loss_meter.meters['loss'].avg.item(), 2) == 4.6
        shutil.rmtree(self.epoch_trainer.work_dir)

    def test_iter_base(self):
        self.iter_trainer.fit([self.data_loader], [('train', 2)])
        loss_meter = self.iter_trainer.loss_meters['TensorboardLoggerHook']

        assert round(loss_meter.meters['loss'].avg.item(), 2) == 8.6
        shutil.rmtree(self.work_dir)
        shutil.rmtree(self.iter_trainer.work_dir)
