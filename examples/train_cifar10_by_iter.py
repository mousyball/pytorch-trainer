import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR

from pytorch_trainer.utils import get_cfg_defaults
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
    testset = torchvision.datasets.CIFAR10(root='./dev/data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return testloader, classes


if __name__ == "__main__":
    # load config
    torch.cuda.empty_cache()

    config = get_cfg_defaults()
    config.merge_from_file('configs/pytorch_trainer/trainer.yaml')
    config.merge_from_list(['HOOK.CheckpointHook.interval', 3])

    # model
    model = Net()

    # dataloader
    val_loader, classes = dataloader()
    train_loader = val_loader

    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD([{'params': model.get_1x_lr()},
                           {'params': model.get_10x_lr(), 'lr': 1e-2}
                           ], lr=1e-3, momentum=0.9)

    # scheduler
    scheduler = StepLR(optimizer, step_size=5, gamma=0.2)

    # initial trainer
    trainer = IterBasedTrainer(model,
                               optimizer=optimizer,
                               scheduler=scheduler,
                               work_dir='./dev/trainer/',
                               logger=None,
                               meta={'commit': 'as65sadf45'},
                               max_iter=15000)

    # register all callback
    trainer.register_callback(config)

    # training: demo will train 15K iteration, run validation 3 time and save 1 weight
    trainer.fit(data_loaders=[train_loader, val_loader],
                workflow=[('train', 5000), ('val', -1)])
