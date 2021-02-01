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

from pytorch_trainer.trainer.epoch_based_trainer import EpochBasedTrainer


class config:
    def __init__(self,
                 optimizer_config=None,
                 scheduler_config=None,
                 checkpoint_config=None,
                 log_config=None):
        self.optimizer_config = optimizer_config  # dict(interval=1)
        # dict(mode='other', interval=1)
        self.scheduler_config = scheduler_config
        # dict(interval=3, save_optimizer=True)
        self.checkpoint_config = checkpoint_config
        self.log_config = log_config  # 'default'


@pytest.mark.parametrize("cfg, expected",
                         [(config(), -1.65),
                          (config(optimizer_config=dict(interval=2)), -0.43)])
def test_optimzier_hook(cfg, expected):
    torch.manual_seed(0)
    model = Model()
    data_loader = DataLoader(torch.ones((5, 2)))

    trainer = EpochBasedTrainer(model,
                                optimizer=optim.SGD(
                                    model.parameters(), lr=0.02),
                                max_epoch=5)
    trainer.register_callback(cfg)
    trainer.fit([data_loader], [('train', 1)])
    assert round(trainer.outputs['loss'].item(), 2) == expected


@pytest.mark.parametrize("cfg, expected",
                         [(config(), 0.004),
                          (config(scheduler_config=dict(interval=2, mode='step')), 0.0008)])
def test_scheduler_hook(cfg, expected):
    torch.manual_seed(0)
    model = Model()
    data_loader = DataLoader(torch.ones((5, 2)))

    optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.2)

    trainer = EpochBasedTrainer(model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                max_epoch=2)
    trainer.register_callback(cfg)
    trainer.fit([data_loader], [('train', 1)])
    assert trainer.optimizer.param_groups[0]['lr'] == expected


@pytest.mark.parametrize("cfg, expected",
                         [(config(checkpoint_config=dict(interval=2)), 'epoch_002.pth')])
def test_checkpoint_hook(cfg, expected):
    work_dir = tempfile.mkdtemp()
    model = Model()
    data_loader = DataLoader(torch.ones((5, 2)))

    optimizer = torch.optim.SGD(model.parameters(), lr=0.02)

    trainer = EpochBasedTrainer(model,
                                optimizer=optimizer,
                                work_dir=work_dir,
                                max_epoch=2)
    trainer.register_callback(cfg)
    trainer.fit([data_loader], [('train', 1)])
    output_path = osp.join(work_dir, 'checkpoint', '*.pth')

    assert osp.basename(glob(output_path)[0]) == expected

    shutil.rmtree(work_dir)


# TODO: logger hook test


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
