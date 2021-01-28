from .builder import LOSSES


class ILoss:
    def __init__(self, **kwargs):
        raise NotImplementedError()

    def __call__(self, output, label):
        raise NotImplementedError()


class BaseLoss(ILoss):
    def __init__(self, **kwargs):
        """Parse parameters from config.

        Args:
            cfg (:obj:`CfgNode`): config dictionary
        """
        raise NotImplementedError()

    def __call__(self, output, label):
        """Self-defined loss calculation.

        Args:
            output (torch.Tensor): model prediction
            label (torch.Tensor): ground truth
        """
        raise NotImplementedError()


@LOSSES.register()
class MyLeNetLoss(BaseLoss):
    def __init__(self, **kwargs):
        from torch.nn import CrossEntropyLoss
        self.__dict__ = kwargs
        self.criterion = CrossEntropyLoss(**kwargs)

    def __call__(self, output, label):
        return self.criterion(output, label)
