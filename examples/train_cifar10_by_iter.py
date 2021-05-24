import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms

from networks.loss.builder import build_loss
from pytorch_trainer.utils import get_cfg_defaults
from pytorch_trainer.trainer.utils import (
    get_device, get_logger, create_work_dir, set_random_seed
)
from pytorch_trainer.optimizers.builder import build_optimizer
from pytorch_trainer.schedulers.builder import build_scheduler
from pytorch_trainer.trainer.iter_based_trainer import IterBasedTrainer


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.criterion = nn.CrossEntropyLoss()

    def get_1x_lr(self):
        for name, param in model.named_parameters():
            if 'conv' in name:
                yield param

    def get_10x_lr(self):
        for name, param in model.named_parameters():
            if 'fc' in name:
                yield param

    def train_step(self, batch_data):
        '''
        box_loss is a dummy loss for multiple loss demo purpose
        '''
        inputs, labels = batch_data

        # forward
        outputs = self(inputs)
        losses = self.criterion(outputs, labels)

        return dict(cls_loss=losses, box_loss=losses+10.0)

    def val_step(self, batch_data):
        inputs, labels = batch_data

        # forward
        outputs = self(inputs)
        losses = criterion(outputs, labels)

        return dict(cls_loss=losses, box_loss=losses+10.0)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def dataloader():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # CIFAR10 train set 25K, val 5K. Use val set for demo purpose
    test_set = torchvision.datasets.CIFAR10(root='./dev/data', train=False,
                                            download=True, transform=transform)
    val_loader = torch.utils.data.DataLoader(test_set, batch_size=4,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return val_loader, classes


if __name__ == "__main__":
    # load config
    config = get_cfg_defaults()
    config.merge_from_file('configs/pytorch_trainer/trainer.yaml')
    config.merge_from_list(['HOOK.CheckpointHook.interval', 3])

    # create working directory
    work_dir = './dev/trainer/'
    work_dir = create_work_dir(work_dir)

    # logger
    logger = get_logger(work_dir)

    # set random seed
    seed = config.trainer.seed
    deterministic = config.trainer.deterministic
    set_random_seed(logger,
                    seed=seed,
                    deterministic=deterministic)

    # get device
    gpu_ids = config.trainer.gpu_ids
    device = get_device(logger,
                        gpu_ids=gpu_ids,
                        deterministic=deterministic)

    # model
    model = Net()

    # dataloader
    val_loader, classes = dataloader()
    train_loader = val_loader

    # loss
    criterion = build_loss(config.loss)

    # optimizer
    lr = config.optimizer.params.lr
    params_group = [{'params': model.get_1x_lr(), 'lr': lr * 1},
                    {'params': model.get_10x_lr(), 'lr': lr * 10}]
    optimizer = build_optimizer(params_group, config.optimizer)

    # scheduler
    scheduler = build_scheduler(optimizer, config.scheduler)

    # initial trainer
    trainer = IterBasedTrainer(model,
                               optimizer=optimizer,
                               scheduler=scheduler,
                               device=device,
                               work_dir=work_dir,
                               logger=logger,
                               meta={'commit': 'as65sadf45'},
                               max_iter=15000)

    # register all callback
    trainer.register_callback(config)

    # training: demo will train 15K iteration, run validation 3 time and save 1 weight
    trainer.fit(data_loaders=[train_loader, val_loader],
                workflow=[('train', 5000), ('val', -1)])
