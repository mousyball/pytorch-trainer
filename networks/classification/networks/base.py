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
        if 'backbone' not in cfg:
            raise KeyError("Key 'backbone' is not in config.")
        self.backbone = build_backbone(cfg.backbone)
        self.criterion = build_loss(cfg.loss)

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

    def get_optimizer_params(self, group_info, lr):
        """Get optimizer parameters from config.

        Args:
            group_info (Tuple(List, Union[int, float])): This contains
                group_list that sends to the optimizer and corresponding
                weight for scaling.
            lr (float): learning rate

        Returns:
            List: parameters group for optimizer.
        """
        # Check if config name matches the attribute name.
        for groups in group_info:
            group_list = groups[0]
            for group in group_list:
                if group not in dir(self):
                    assert False, f"{group} not in {self.__dict__.keys()}"

        params_group = []
        for group_list, weight in group_info:
            params_group.append(
                {
                    'params': self.get_lr_params(group_list),
                    'lr': lr * weight
                }
            )

        return params_group

    def train_step(self, batch_data):
        """Define training step."""
        inputs, labels = batch_data['inputs'], batch_data['targets']
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, labels)

        return dict(cls_loss=loss)

    def val_step(self, batch_data):
        """Define validation step."""
        inputs, labels = batch_data['inputs'], batch_data['targets']
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, labels)

        return dict(cls_loss=loss)

    def forward(self, x):
        """Define forward propagation."""
        output = self.backbone(x)
        return output
