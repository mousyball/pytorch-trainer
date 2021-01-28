from pytorch_trainer.utils.builder import build
from pytorch_trainer.utils.registry import Registry

LOSSES = Registry('loss')


def build_loss(cfg):
    return build(cfg, LOSSES)
