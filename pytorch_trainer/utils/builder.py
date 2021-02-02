from copy import deepcopy


def build(cfg, registry):
    """Build a network.

    Args:
        cfg (dict): The config of network.
        registry (:obj:`Registry`): A registry the module belongs to.

    Returns:
        nn.Module: A built nn module.
    """
    _cfg = deepcopy(cfg)

    obj_name = _cfg.pop('NAME')
    assert isinstance(obj_name, str)

    # [NOTE] 'KeyError' is handled in registry.
    obj_cls = registry.get(obj_name)

    # [Case]: LOSS
    if registry._name == 'loss':
        return obj_cls(**dict(_cfg))

    return obj_cls(_cfg)
