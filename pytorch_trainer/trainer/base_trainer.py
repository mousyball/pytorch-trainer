"""
File is modified from https://github.com/open-mmlab/mmcv
License File Available at:
https://github.com/open-mmlab/mmcv/blob/master/LICENSE
"""

from .utils import sync_counter
from ..utils import Registry
from .priority import get_priority
from .log_meter import LossMeter
from .hooks.base_hook import HOOKS

TRAINER = Registry('trainer')


class BaseTrainer():
    def __init__(self,
                 model,
                 max_iter=0,
                 max_epoch=0,
                 optimizer=None,
                 scheduler=None,
                 device=None,
                 work_dir=None,
                 logger=None,
                 meta=None
                 ):
        """The base class of Trainer, a training helper for PyTorch.
        All subclasses should implement the following APIs:
        - ``fit()``
        - ``train()``
        - ``val()``

        Args:
            model (:obj:`torch.nn.Module`): The model to be run.
            max_epoch (int): Total training epochs.
            optimizer (dict or `torch.optim.Optimizer`): It can be either an
                optimizer (in most cases) or a dict of optimizers (in models that
                requires more than one optimizer, e.g., GAN).
            work_dir (str, optional): The working directory to save checkpoints
                and logs. Defaults to None.
            logger (`logging.Logger`): Logger used during training. Defaults to None.
            meta (dict, optional): A dict records meta data. Defaults to None.
        """

        # TODO: argument checker
        # TODO: discuss meta format for logging
        # TODO: Handle optimizer, scheduler, device by config
        self.meta = meta
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        self.work_dir = work_dir
        self.device = device

        # get model name from the model class
        if hasattr(self.model, 'module'):
            self._model_name = self.model.module.__class__.__name__
        else:
            self._model_name = self.model.__class__.__name__

        self.mode = None
        self._hooks = []
        self._epoch = 0
        self._max_epoch = max_epoch
        self._iter = 0
        self._inner_iter = 0
        self._max_iter = max_iter
        self.loss_meters = dict()

    @property
    def iter(self):
        return self._iter

    @property
    def inner_iter(self):
        return self._inner_iter

    @property
    def max_iter(self):
        return self._max_iter

    @property
    def epoch(self):
        return self._epoch

    @property
    def max_epoch(self):
        return self._max_epoch

    def _loss_parser(self, output):
        """Sum up the losses of output.

        Args:
            output (dict): example dict(cls_loss=float, regr_loss=float)
        Returns:
            dict: dictionary include loss(total loss) and multi_loss(rest of losses)
        """
        total_loss = 0
        for value in output.values():
            total_loss += value

        return dict(loss=total_loss,
                    multi_loss=output)

    def call_hook(self, fn_name):
        """Call all hooks by name.
        Args:
            fn_name (str): The function name in each hook to be called, such as
                "before_train_epoch".
        """
        for hook in self._hooks:
            getattr(hook, fn_name)(self)

    @sync_counter
    def call_hook_with_sync(self, fn_name):
        """sync iteration and epoch version of call_hook"""
        self.call_hook(fn_name)

    def register_hook(self, hook, priority='NORMAL'):
        """Register a hook into the hook list.
        The hook will be inserted into a priority queue, with the specified
        priority (See :class:`Priority` for details of priorities).
        For hooks with the same priority, they will be triggered in the same
        order as they are registered.
        Args:
            hook (:obj:`Hook`): The hook to be registered.
            priority (int or str or :obj:`Priority`): Hook priority.
                Lower value means higher priority.

        """
        # assert isinstance(hook(), Hook)

        hook.priority = get_priority(priority)
        # insert the hook to a sorted list
        for i in range(len(self._hooks) - 1, -1, -1):
            if hook.priority >= self._hooks[i].priority:
                self._hooks.insert(i + 1, hook)
                return
        self._hooks.insert(0, hook)

    def _register_hook(self, config):
        for name in config.get('name'):
            # get arguments
            kwargs = config.get(name)
            priority = kwargs.pop('priority')

            # get function
            log_hook = HOOKS.get(name)(**kwargs)
            self.register_hook(log_hook, priority=priority)

    def _register_loss_meter(self, config):
        for key in config.name:
            self.loss_meters[key] = LossMeter()

    def register_callback(self,
                          config):
        """Register hooks for training.
            append hook into list self.hooks
        """
        self._register_loss_meter(config.logger_hook)
        self._register_hook(config.logger_hook)
        self._register_hook(config.hook)
        # self._register_hook(config.CUSTOM_HOOK)

    def data_to_device(self, data):
        # TODO: place to hook
        if isinstance(data, dict):
            return {k: v.to(self.device) for k, v in data.items()}
        else:
            inputs, labels = data
            return inputs.to(self.device), labels.to(self.device)
