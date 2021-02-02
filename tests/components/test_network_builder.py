import os.path as osp

import torch
import pytest
import torch.nn as nn

from pytorch_trainer.utils.config import parse_yaml_config
from networks.classification.builder import build_network

ROOT_PATH = './configs/networks/classification/'


class TestClassification:
    @pytest.mark.parametrize("filename", ['lenet.yaml', 'mynet.yaml'])
    def test_network_builder_with_cfg(self, filename):
        """Network tests include backbone/loss builder
        """
        FILENAME = filename
        cfg = parse_yaml_config(osp.join(ROOT_PATH, FILENAME))
        net = build_network(cfg)
        assert isinstance(net, nn.Module)

    def test_network_builder_with_keyerror(self):
        FILENAME = 'lenet.yaml'
        cfg = parse_yaml_config(osp.join(ROOT_PATH, FILENAME))
        # Remove keyword: 'NETWORK'
        cfg.pop('NETWORK')
        with pytest.raises(KeyError) as excinfo:
            _ = build_network(cfg)
        assert "KeyError" in str(excinfo)

    @pytest.mark.parametrize("filename", ['lenet.yaml', 'mynet.yaml'])
    def test_network_forward(self, filename):
        FILENAME = filename
        # Construct network
        cfg = parse_yaml_config(osp.join(ROOT_PATH, FILENAME))
        net = build_network(cfg)
        net.eval()
        # Initialize input
        if cfg.get('NETWORK'):
            n_class = cfg.get('NETWORK').get('BACKBONE').get('NUM_CLASS')
        elif cfg.get('CUSTOM'):
            n_class = cfg.get('CUSTOM').get('MODEL').get('NUM_CLASS')
        else:
            assert False
        x = torch.rand(4, 3, 32, 32)
        # Inference
        output_size = net(x).shape
        assert output_size == torch.Size([4, n_class])

    @pytest.mark.parametrize("filename", ['lenet.yaml', 'mynet.yaml'])
    def test_network_train_step(self, filename):
        FILENAME = filename
        # Construct network
        cfg = parse_yaml_config(osp.join(ROOT_PATH, FILENAME))
        net = build_network(cfg)
        net.train()
        # Initialize input
        if cfg.get('NETWORK'):
            n_class = cfg.get('NETWORK').get('BACKBONE').get('NUM_CLASS')
        elif cfg.get('CUSTOM'):
            n_class = cfg.get('CUSTOM').get('MODEL').get('NUM_CLASS')
        else:
            assert False
        x = torch.rand(4, 3, 32, 32)
        y = torch.randint(low=0, high=n_class, size=(4,))
        # Training Step
        output_loss = net.train_step((x, y))
        assert 'loss' in output_loss
        assert 'multi_loss' in output_loss

    @pytest.mark.parametrize("filename", ['lenet.yaml', 'mynet.yaml'])
    def test_network_val_step(self, filename):
        FILENAME = filename
        # Construct network
        cfg = parse_yaml_config(osp.join(ROOT_PATH, FILENAME))
        net = build_network(cfg)
        net.eval()
        # Initialize input
        if cfg.get('NETWORK'):
            n_class = cfg.get('NETWORK').get('BACKBONE').get('NUM_CLASS')
        elif cfg.get('CUSTOM'):
            n_class = cfg.get('CUSTOM').get('MODEL').get('NUM_CLASS')
        else:
            assert False
        x = torch.rand(4, 3, 32, 32)
        y = torch.randint(low=0, high=n_class, size=(4,))
        # Training Step
        output_loss = net.val_step((x, y))
        assert 'loss' in output_loss
        assert 'multi_loss' in output_loss

    def test_backbone_builder_with_keyerror(self):
        FILENAME = 'lenet.yaml'
        cfg = parse_yaml_config(osp.join(ROOT_PATH, FILENAME))
        # Remove keyword: 'BACKBONE'
        cfg.NETWORK.pop('BACKBONE')
        with pytest.raises(KeyError) as excinfo:
            _ = build_network(cfg)
        assert "KeyError" in str(excinfo)

    def test_utils_builder_with_keyerror(self):
        FILENAME = 'lenet.yaml'
        cfg = parse_yaml_config(osp.join(ROOT_PATH, FILENAME))
        # Modify keyword name
        cfg.defrost()
        cfg.NETWORK.BACKBONE.NAME = 'WrongLeNet'
        with pytest.raises(KeyError) as excinfo:
            _ = build_network(cfg)
        assert "KeyError" in str(excinfo)