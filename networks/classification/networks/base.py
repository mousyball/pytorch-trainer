import torch.nn as nn

from ...loss import build_loss
from ..builder import build_backbone


class INetwork(nn.Module):
    def __init__(self):
        super(INetwork, self).__init__()

    def _construct_network(self, cfg):
        """Construct network from builder."""
        raise NotImplementedError()

    def freeze(self):
        """Freeze components or layers."""
        raise NotImplementedError()

    def get_lr_params(self, group_list):
        """Get LR group for optimizer."""
        raise NotImplementedError()

    def train_step(self, batch_data):
        """Define training step."""
        raise NotImplementedError()

    def val_step(self, batch_data):
        """Define validation step."""
        raise NotImplementedError()

    def forward(self, x):
        """Define forward propagation."""
        raise NotImplementedError()


class BaseNetwork(INetwork):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def _construct_network(self, cfg):
        """Construct network from builder."""
        if 'BACKBONE' not in cfg:
            raise KeyError("Key 'BACKBONE' is not in config.")
        self.backbone = build_backbone(cfg.BACKBONE)
        self.criterion = build_loss(cfg.LOSS)

    def freeze(self):
        """Freeze components or layers.
        TODO:
            * Freeze all or backbone.
        NOTE:
            * Freeze layers depended on demand.
        """
        raise NotImplementedError()

    def get_lr_params(self, group_list):
        """Get LR group for optimizer.
        TODO:
            * Checker for this function.
            * Config yaml for components.
            * Any better coding logic?
        NOTE:
            * Make sure that the following layers exist in the network!!!
              * For example, `nn.Conv1d` is used in network and you should add it to the condition below.
            * The parameters wouldn't be updated if they are freezed in advance.
        """
        modules = [self.__getattr__(m) for m in group_list]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) \
                        or isinstance(m[1], nn.Conv1d) \
                        or isinstance(m[1], nn.Linear) \
                        or isinstance(m[1], nn.BatchNorm2d) \
                        or isinstance(m[1], nn.SyncBatchNorm) \
                        or isinstance(m[1], nn.GroupNorm):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def train_step(self, batch_data):
        """Define training step."""
        inputs, labels = batch_data
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, labels)

        return dict(loss=loss,
                    multi_loss=dict(cls_loss=loss))

    def val_step(self, batch_data):
        """Define validation step."""
        inputs, labels = batch_data
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, labels)

        return dict(loss=loss,
                    multi_loss=dict(cls_loss=loss))

    def forward(self, x):
        """Define forward propagation."""
        output = self.backbone(x)
        return output
