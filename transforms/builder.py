from copy import deepcopy

from pytorch_trainer.utils.registry import Registry

TRANSFORMS = Registry('Transform')


def build(cfg, registry):
    """Build a single transform object.

    Args:
        cfg (dict): The config of transform object.
        registry (:obj:`Registry`): A registry the module belongs to.

    Returns:
        object: A built class.
    """
    _cfg = deepcopy(cfg)

    obj_name = _cfg.get('name')
    assert isinstance(obj_name, str)

    params = dict() if _cfg.get('params') is None else cfg.params
    input_key = _cfg.get('input_key')
    output_key = _cfg.get('output_key')
    visualize = _cfg.get('visualize')

    obj_cls = registry.get(obj_name)

    return obj_cls(**params,
                   input_key=input_key,
                   output_key=output_key,
                   visualize=visualize)


def build_transform(cfg):
    return build(cfg, TRANSFORMS)
