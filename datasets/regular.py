from torchvision import datasets
from torch.utils.data import DataLoader

from .builder import DATASETS, DATALOADERS

# [NOTE] Pytorch official dataset API
torch_dataset = {
    'cifar10': datasets.CIFAR10,
}
torch_dataloader = {
    'dataloader': DataLoader,
}


for k, v in torch_dataset.items():
    DATASETS._do_register(k, v)

for k, v in torch_dataloader.items():
    DATALOADERS._do_register(k, v)
