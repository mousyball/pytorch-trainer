from fvcore.common.config import CfgNode as CN

# [NOTE] Default field is free to add any node.
# Because this base config could be shared among applications, it should be clean to any import.
_C = CN(new_allowed=True)


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()


def parse_yaml_config(config_path, allow_unsafe=False):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(config_path, allow_unsafe)
    cfg.freeze()
    return cfg
