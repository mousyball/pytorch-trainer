from .base_logger import LoggerHook

# TODO: registry
class TextLoggerHook(LoggerHook):
    def __init__(self, interval=10):
        super().__init__(interval=interval)
