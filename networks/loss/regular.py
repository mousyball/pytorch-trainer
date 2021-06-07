import torch.nn as nn

from .builder import LOSSES

# [NOTE] Pytorch official API
torch_loss = {
    'CrossEntropyLoss': nn.CrossEntropyLoss,
    'BCEWithLogitsLoss': nn.BCEWithLogitsLoss
}

for k, v in torch_loss.items():
    LOSSES._do_register(k, v)
