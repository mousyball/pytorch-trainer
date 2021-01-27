import torch.nn as nn

from .base import BaseNetwork
from ..builder import NETWORKS


@NETWORKS.register()
class LeNet(BaseNetwork):
    """
    TODO:
        * double instantiation: super, here
        * Remove criterion after loss builder is ready.
    """

    def __init__(self, cfg, **kwargs):
        super(LeNet, self).__init__()
        self._construct_network(cfg)
        self.criterion = nn.CrossEntropyLoss()
