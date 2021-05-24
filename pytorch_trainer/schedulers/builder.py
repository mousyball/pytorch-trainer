from pytorch_trainer.utils.builder import build
from pytorch_trainer.utils.registry import Registry

SCHEDULERS = Registry('scheduler')


def build_scheduler(optimizer, cfg):
    return build(cfg, SCHEDULERS, optimizer=optimizer)
