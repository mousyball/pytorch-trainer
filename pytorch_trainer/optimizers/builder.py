from pytorch_trainer.utils.builder import build
from pytorch_trainer.utils.registry import Registry

OPTIMIZERS = Registry('optimizer')


def build_optimizer(params_group, cfg):
    return build(cfg, OPTIMIZERS, params_group=params_group)
