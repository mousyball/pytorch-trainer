from typing import Any

from fvcore.common.registry import Registry as _Registry


class Registry(_Registry):
    def register(self, obj: Any = None, name=None) -> Any:
        """
        Register the given object under the the name `obj.__name__` or given name.
        Can be used as either a decorator or not. See docstring of this class for usage.

        NOTE:
            * Add a naming argument for registry. It's same feature as registry of mmcv.
        """
        if obj is None:
            # used as a decorator
            def deco(func_or_class: Any) -> Any:
                _name = func_or_class.__name__ if name is None else name
                self._do_register(_name, func_or_class)
                return func_or_class

            return deco

        # used as a function call
        _name = obj.__name__ if name is None else name
        self._do_register(_name, obj)
