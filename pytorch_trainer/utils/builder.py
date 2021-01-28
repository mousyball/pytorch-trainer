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
    obj_name = _cfg.get('NAME')

    if isinstance(obj_name, str):
        obj_cls = registry.get(obj_name)
        if obj_cls is None:
            raise KeyError(
                f'{obj_name} is not in the {registry._name} registry')
    else:
        raise TypeError(
            f'type must be a str, but got {type(obj_name)}')

    # [Case]: LOSS
    if registry._name == 'loss':
        _cfg.pop('NAME')
        return obj_cls(**dict(_cfg))

    return obj_cls(_cfg)
