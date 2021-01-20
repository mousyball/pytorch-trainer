from ..base_hook import HOOKS
from .base_logger import LoggerHook


@HOOKS.register_module()
class TextLoggerHook(LoggerHook):
    def __init__(self, interval=10):
        super().__init__(interval=interval)
