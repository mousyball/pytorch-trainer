from pytorch_trainer.utils.registry import Registry

BACKBONES = Registry('backbone')
NETWORKS = Registry('network')
CUSTOMS = Registry('custom')


def build(cfg, registry):
    """Build a network.

    Args:
        cfg (dict): The config of network.
        registry (:obj:`Registry`): A registry the module belongs to.

    Returns:
        nn.Module: A built nn module.
    """
    obj_name = cfg.get('NAME')

    if isinstance(obj_name, str):
        obj_cls = registry.get(obj_name)
        if obj_cls is None:
            raise KeyError(
                f'{obj_name} is not in the {registry.name} registry')
    else:
        raise TypeError(
            f'type must be a str, but got {type(obj_name)}')

    return obj_cls(cfg)


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
