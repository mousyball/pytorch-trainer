import torch
import pytest
import torch.nn as nn
from torch.utils.data import DataLoader

from pytorch_trainer.trainer import IterBasedTrainer, EpochBasedTrainer
from pytorch_trainer.trainer.hooks import (
    OptimizerHook, SchedulerHook, CheckpointHook
)
from pytorch_trainer.trainer.base_trainer import BaseTrainer


class config:
    def __init__(self,
                 optimizer_config=None,
                 scheduler_config=None,
                 checkpoint_config=None,
                 log_config=None):
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.checkpoint_config = checkpoint_config
        self.log_config = log_config  # 'default'


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.linear.weight.data.fill_(0.5)
        self.linear.bias.data.fill_(0.1)

    def forward(self, x):
        return self.linear(x)

    def train_step(self, x):
        return dict(loss=self(x))

    def val_step(self, x):
        return dict(loss=self(x))


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
    model = Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
    trainer = EpochBasedTrainer(model,
                                optimizer=optimizer,
                                max_epoch=1)
    trainer.register_callback(config(optimizer_config=dict(interval=2)))
    data_loader = DataLoader(torch.ones((1, 2)))

    def test_val(self):
        self.trainer.fit([self.data_loader, self.data_loader], [
                         ('train', 1), ('val', 1)])

        assert round(self.trainer.outputs['loss'].item(), 2) == 1.1

    def test_flow(self):
        self.trainer.fit([self.data_loader], [('train', 2)])

        assert self.trainer.epoch == 1


class Test_iter_based_trainer:
    model = Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
    trainer = IterBasedTrainer(model,
                               optimizer=optimizer,
                               max_iter=1)
    trainer.register_callback(config(optimizer_config=dict(interval=2)))
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


class Test_register_callback():

    @pytest.mark.parametrize('cfg', [
        config()
    ])
    def test_regist_optimizer_hook(self, cfg):
        model = Model()
        trainer = EpochBasedTrainer(model)

        cfg = config()
        trainer.register_callback(cfg)
        for hook in trainer._hooks:
            if isinstance(hook, OptimizerHook):
                assert True and len(trainer._hooks) == 1
                return
            else:
                continue
        assert False

    @pytest.mark.parametrize('cfg', [
        config(),
        config(scheduler_config=dict(interval=2))
    ])
    def test_regist_scheduler_hook(self, cfg):
        model = Model()
        trainer = EpochBasedTrainer(model,
                                    scheduler='not_none')

        trainer.register_callback(cfg)
        for hook in trainer._hooks:
            if isinstance(hook, SchedulerHook):
                assert True and len(trainer._hooks) == 2
                return
            else:
                continue
        assert False

    @pytest.mark.parametrize('cfg', [
        config(checkpoint_config=dict(interval=2))
    ])
    def test_regist_checkpoint_hook(self, cfg):
        model = Model()
        trainer = EpochBasedTrainer(model)

        trainer.register_callback(cfg)
        for hook in trainer._hooks:
            if isinstance(hook, CheckpointHook):
                assert True and len(trainer._hooks) == 2
                return
            else:
                continue
        assert False

    @pytest.mark.parametrize('scheduler, expected', [
        (None, 2),
        ('scheduler', 3),
    ])
    def test_regist_all(self, scheduler, expected):
        model = Model()
        trainer = EpochBasedTrainer(model,
                                    scheduler=scheduler)

        cfg = config(optimizer_config=dict(interval=1),
                     scheduler_config=dict(mode='other', interval=1),
                     checkpoint_config=dict(interval=3, save_optimizer=True))
        trainer.register_callback(cfg)

        assert len(trainer._hooks) == expected

    # TODO: logger test
