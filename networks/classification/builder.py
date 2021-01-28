from pytorch_trainer.utils.builder import build
from pytorch_trainer.utils.registry import Registry

BACKBONES = Registry('backbone')
NETWORKS = Registry('network')
CUSTOMS = Registry('custom')


def build_backbone(cfg):
    return build(cfg, BACKBONES)


def build_network(cfg):
    """Build network

    Args:
        cfg (dict): The config of network.

    Raises:
        KeyError: Check if key in the config.

    Returns:
        nn.Module: built nn module.
    """
    key = 'CUSTOM'
    _cfg = cfg.get(key)
    if _cfg is not None:
        return build(_cfg, CUSTOMS)

    key = 'NETWORK'
    _cfg = cfg.get(key)
    if _cfg is None:
        raise KeyError(f"Key '{key}' is not in config.")
    return build(_cfg, NETWORKS)
