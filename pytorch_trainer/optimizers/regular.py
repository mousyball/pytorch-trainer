import torch.optim as optim

from .builder import OPTIMIZERS

# [NOTE] Pytorch official API
torch_optimizer = {
    'SGD': optim.SGD,
    'Adam': optim.Adam,
    'AdamW': optim.AdamW
}

for k, v in torch_optimizer.items():
    OPTIMIZERS._do_register(k, v)
