import shutil
import os.path as osp

import torch
import pytest

from pytorch_trainer.utils.config import parse_yaml_config
from pytorch_trainer.trainer.utils import (
    get_logger, create_work_dir, load_pretained_weight
)
from networks.classification.builder import build_network

CFG_PATH = './configs/pytorch_trainer/iter_trainer.yaml'
WORK_DIR = './dev/tests/'


class TestLoadPretraintedWeight:
    def test_load_pretrained_weight(self):
        # load config
        config = parse_yaml_config(CFG_PATH)

        # create working directory
        work_dir = './dev/tests/pretrained_weights'
        work_dir = create_work_dir(work_dir)

        # logger
        logger = get_logger(work_dir)

        # model
        model = build_network(config)

        # [NOTE] Save the random weight first, then reload for checking.
        output_model_name = 'random_cifar10.pth'
        output_path = osp.join(work_dir, output_model_name)
        torch.save(model.state_dict(), output_path)

        # load pretrained weight
        if config.trainer.pretrain.load_weight:
            dir_path = work_dir
            weight_name = output_model_name
            load_pretained_weight(model,
                                  dir_path,
                                  weight_name,
                                  logger)

    @pytest.mark.xfail(reason="Load an incompleted weight.")
    def test_load_pretrained_weight_xfail(self):
        # load config
        config = parse_yaml_config(CFG_PATH)

        # create working directory
        work_dir = './dev/tests/pretrained_weights'
        work_dir = create_work_dir(work_dir)

        # logger
        logger = get_logger(work_dir)

        # model
        model = build_network(config)

        # [NOTE] Save the random weight first, then reload for checking.
        output_model_name = 'random_cifar10.pth'
        output_path = osp.join(work_dir, output_model_name)
        # [NOTE] Delete weights and save it.
        del model.backbone.fc3.weight
        del model.backbone.fc3.bias
        torch.save(model.state_dict(), output_path)

        # Rebuild the model
        model = build_network(config)

        # load pretrained weight
        if config.trainer.pretrain.load_weight:
            dir_path = work_dir
            weight_name = output_model_name
            load_pretained_weight(model,
                                  dir_path,
                                  weight_name,
                                  logger)

    shutil.rmtree(WORK_DIR)
