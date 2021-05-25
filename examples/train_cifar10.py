import argparse

import torch
import torchvision
import torchvision.transforms as transforms

from pytorch_trainer.trainer import build_trainer_api


def argparser():
    parser = argparse.ArgumentParser(
        description='Trainer demo',
        formatter_class=lambda prog: argparse.HelpFormatter(
            prog,
            max_help_position=40
        )
    )

    parser.add_argument('-cfg',
                        '--config',
                        default='configs/pytorch_trainer/epoch_trainer.yaml',
                        type=str,
                        metavar='PATH',
                        help=r'config path')

    return parser.parse_args()


def dataloader():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # CIFAR10 train set 25K, val 5K. Use val set for demo purpose
    test_set = torchvision.datasets.CIFAR10(root='./dev/data', train=False,
                                            download=True, transform=transform)
    val_loader = torch.utils.data.DataLoader(test_set, batch_size=2,
                                             shuffle=False, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return val_loader, classes


if __name__ == "__main__":
    parser = argparser()
    trainer, workflow = build_trainer_api(parser.config)

    # dataloader
    val_loader, classes = dataloader()
    train_loader = val_loader

    # training: demo will train 15K iteration(3 epoch), run validation 3 time and save 3 weight
    trainer.fit(data_loaders=[train_loader, val_loader],
                workflow=workflow)
