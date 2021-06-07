from .base import BaseNetwork
from ..builder import NETWORKS


@NETWORKS.register()
class LeNet(BaseNetwork):
    def __init__(self, cfg):
        super(LeNet, self).__init__()
        self._construct_network(cfg)
