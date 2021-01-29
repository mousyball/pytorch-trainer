import pytest
import torch.nn as nn

from pytorch_trainer.trainer.hooks import (
    OptimizerHook, SchedulerHook, CheckpointHook
)
from pytorch_trainer.trainer.epoch_based_trainer import EpochBsedTrainer


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


class config:
    def __init__(self,
                 optimizer_config=None,
                 scheduler_config=None,
                 checkpoint_config=None,
                 log_config=None):
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.checkpoint_config = checkpoint_config
        self.log_config = log_config


class Test_register_callback():

    @pytest.mark.parametrize('cfg', [
        config()
    ])
    def test_regist_optimizer_hook(self, cfg):
        model = Model()
        trainer = EpochBsedTrainer(model)

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
        trainer = EpochBsedTrainer(model,
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
        trainer = EpochBsedTrainer(model)

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
        trainer = EpochBsedTrainer(model,
                                   scheduler=scheduler)

        cfg = config(optimizer_config=dict(interval=1),
                     scheduler_config=dict(mode='other', interval=1),
                     checkpoint_config=dict(interval=3, save_optimizer=True))
        trainer.register_callback(cfg)

        assert len(trainer._hooks) == expected

    # TODO: logger test
