import torch.nn as nn


class IBackbone(nn.Module):
    def __init__(self):
        super(IBackbone, self).__init__()

    def init_weights(self):
        """Initialize the weights in your network."""
        raise NotImplementedError()

    def forward(self, x):
        """Define forward propagation."""
        raise NotImplementedError()


class BaseBackbone(IBackbone):
    def __init__(self):
        """[NOTE] Define your network in submodule."""
        super(IBackbone, self).__init__()

    def init_weights(self):
        """Initialize the weights in your network.
        TODO:
            * Any better coding logic?
        NOTE:
            * Check if the following layers match your network.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.SyncBatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        """Define forward propagation."""
        raise NotImplementedError()
