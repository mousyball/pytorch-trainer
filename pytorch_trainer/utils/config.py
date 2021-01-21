from fvcore.common.config import CfgNode as CN

_C = CN()

_C.RUNNER = CN(new_allowed=True)
_C.RUNNER.EPOCH = 100
_C.RUNNER.ITERATION = 1000


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
