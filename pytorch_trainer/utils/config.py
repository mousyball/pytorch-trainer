import os
import logging
from typing import Any, Dict

import yaml
from fvcore.common.config import CfgNode as _CfgNode

BASE_KEY = "_BASE_"


class CfgNode(_CfgNode):
    """Support more features by inheritance.

    NOTE:
        * Support `list` type to `_BASE_` inheritance functionality.
    """

    @classmethod
    def load_yaml_with_base(cls, filename: str, allow_unsafe: bool = False) -> None:
        """
        Just like `yaml.load(open(filename))`, but inherit attributes from its
            `_BASE_`.

        Args:
            filename (str or file-like object): the file name or file of the current config.
                Will be used to find the base config file.
            allow_unsafe (bool): whether to allow loading the config file with
                `yaml.unsafe_load`.

        Returns:
            (dict): the loaded yaml
        """

        with cls._open_cfg(filename) as f:
            try:
                cfg = yaml.safe_load(f)
            except yaml.constructor.ConstructorError:
                if not allow_unsafe:
                    raise
                logger = logging.getLogger(__name__)
                logger.warning(
                    "Loading config {} with yaml.unsafe_load. Your machine may "
                    "be at risk if the file contains malicious content.".format(
                        filename
                    )
                )
                f.close()
                with cls._open_cfg(filename) as f:
                    cfg = yaml.unsafe_load(f)

        # pyre-ignore
        def merge_a_into_b(a: Dict[Any, Any], b: Dict[Any, Any]) -> None:
            # merge dict a into dict b. values in a will overwrite b.
            for k, v in a.items():
                if isinstance(v, dict) and k in b:
                    assert isinstance(
                        b[k], dict
                    ), "Cannot inherit key '{}' from base!".format(k)
                    merge_a_into_b(v, b[k])
                else:
                    b[k] = v

        def _load_base_cfg(base_cfg_file):
            if base_cfg_file.startswith("~"):
                base_cfg_file = os.path.expanduser(base_cfg_file)
            if not any(map(base_cfg_file.startswith, ["/", "https://", "http://"])):
                # the path to base cfg is relative to the config file itself.
                base_cfg_file = os.path.join(
                    os.path.dirname(filename),
                    base_cfg_file)
            base_cfg = cls.load_yaml_with_base(
                base_cfg_file,
                allow_unsafe=allow_unsafe)

            return base_cfg

        if BASE_KEY in cfg:
            BASE_VALUE = cfg.pop(BASE_KEY)
            _base_group = {}
            if isinstance(BASE_VALUE, list):
                # Merge the ancestor yamls first.
                for _BASE_VALUE in BASE_VALUE:
                    base_cfg = _load_base_cfg(_BASE_VALUE)
                    # Merge each base_cfg into _base_group.
                    merge_a_into_b(base_cfg, _base_group)

                # Merge the entry cfg into _base_group at last.
                merge_a_into_b(cfg, _base_group)

                return _base_group
            elif isinstance(BASE_VALUE, str):
                # Load the ancestor yaml first.
                base_cfg = _load_base_cfg(BASE_VALUE)
                # Then, merge the children into the ancestor.
                merge_a_into_b(cfg, base_cfg)
                return base_cfg

        return cfg


# [NOTE] Default field is free to add any node.
# Because this base config could be shared among applications, it should be clean to any import.
_C = CfgNode(new_allowed=True)


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
