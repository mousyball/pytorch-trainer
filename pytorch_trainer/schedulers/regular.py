import torch.optim.lr_scheduler as lr_scheduler

from .builder import SCHEDULERS

# [NOTE] Pytorch official API
torch_scheduler = {
    'StepLR': lr_scheduler.StepLR,
    'CosineAnnealingLR': lr_scheduler.CosineAnnealingLR,
    'CosineAnnealingWarmRestarts': lr_scheduler.CosineAnnealingWarmRestarts
}

for k, v in torch_scheduler.items():
    SCHEDULERS._do_register(k, v)
