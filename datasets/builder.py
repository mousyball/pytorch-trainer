from pytorch_trainer.utils.builder import build
from pytorch_trainer.utils.registry import Registry

DATASETS = Registry('dataset')
DATALOADERS = Registry('dataloader')


def build_dataset(cfg, transform):
    return build(cfg, DATASETS, transform=transform)


def build_dataloader(cfg, transform):
    train_transform, val_transform = transform
    if cfg.dataset.get('train', False):
        train_set = build_dataset(cfg.dataset.train, train_transform)
    if cfg.dataloader.get('train', False):
        train_loader = build(cfg.dataloader.train,
                             DATALOADERS, dataset=train_set)

    if cfg.dataset.get('val', False):
        val_set = build_dataset(cfg.dataset.val, val_transform)
    if cfg.dataloader.get('val', False):
        val_loader = build(cfg.dataloader.val, DATALOADERS, dataset=val_set)

    return train_loader, val_loader
