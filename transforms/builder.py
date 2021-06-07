from copy import deepcopy

from pytorch_trainer.utils.registry import Registry

TRANSFORMS = Registry('Transform')
CUSTOM_TRANSFORMS = Registry('Custom_Transform')


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

    # [NOTE] Custom transform is fully controlled by user.
    if registry == CUSTOM_TRANSFORMS:
        obj_cls = registry.get(obj_name)
        transform = obj_cls().get_train_transforms()
        return transform

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


def build_custom_transform(cfg):
    return build(cfg, CUSTOM_TRANSFORMS)
