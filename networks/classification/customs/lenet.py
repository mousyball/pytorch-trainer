import torch.nn as nn
import torch.nn.functional as F

from ...loss import build_loss
from ..builder import CUSTOMS
from ..networks.base import BaseNetwork


@CUSTOMS.register()
class MyLeNet(BaseNetwork):
    def __init__(self, cfg, **kwargs):
        super(MyLeNet, self).__init__()
        self._construct_network(cfg)

    def _construct_network(self, cfg):
        """Construct network manually.
        NOTE:
            * Overwrite the parent method if needed.
            * Parameters checking isn't involved if customization is utilized.
        """
        self.model = LeNet(cfg.model)
        self.criterion = build_loss(cfg.loss)

    def get_lr_params(self, group_list):
        """Get LR group for optimizer."""
        # [NOTE] Make sure that config matches the network route.
        modules = [self.__getattr__(m) for m in group_list]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) \
                        or isinstance(m[1], nn.Linear):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def forward(self, x):
        """Define forward propagation."""
        output = self.model(x)
        return output


class LeNet(nn.Module):
    def __init__(self, cfg):
        super(LeNet, self).__init__()
        num_class = cfg.get('num_class')

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_class)

    def init_weights(self):
        """Initialize the weights in your network.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
