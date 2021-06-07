from copy import deepcopy


def build(cfg, registry, **kwargs):
    """Build a network.

    Args:
        cfg (dict): The config of network.
        registry (:obj:`Registry`): A registry the module belongs to.

    Returns:
        nn.Module: A built nn module.
    """
    _cfg = deepcopy(cfg)

    obj_name = _cfg.pop('name')
    assert isinstance(obj_name, str)

    # [NOTE] 'KeyError' is handled in registry.
    obj_cls = registry.get(obj_name)

    if registry._name == 'loss':
        return obj_cls(**dict(_cfg))
    elif registry._name == 'optimizer':
        return obj_cls(kwargs['params_group'],
                       **dict(_cfg.params))
    elif registry._name == 'scheduler':
        return obj_cls(kwargs['optimizer'],
                       **dict(_cfg.params))
    elif registry._name in ['trainer', 'dataset', 'dataloader']:
        return obj_cls(**kwargs,
                       **dict(_cfg.params))

    return obj_cls(_cfg)
