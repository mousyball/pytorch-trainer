import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR

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
        loss = self.criterion(outputs, labels)

        return dict(loss=loss,
                    multi_loss=dict(cls_loss=loss))

    def val_step(self, batch_data):
        inputs, labels = batch_data
        # forward
        outputs = self(inputs)
        loss = criterion(outputs, labels)

        return dict(loss=loss,
                    multi_loss=dict(cls_loss=loss))

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

    trainset = torchvision.datasets.CIFAR10(root='./dev/data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./dev/data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes


class dummy_config:
    def __init__(self):
        self.optimizer_config = dict(interval=1)
        self.scheduler_config = dict(mode='other',
                                     interval=1)
        self.checkpoint_config = dict(interval=3,
                                      save_optimizer=True)
        self.log_config = 'default'


if __name__ == "__main__":
    # model
    model = Net()

    # dataloader
    train_loader, val_loader, classes = dataloader()
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
                               max_iter=5000)

    # register all callback
    trainer.register_callback(dummy_config())

    # training
    trainer.fit(data_loaders=[train_loader, val_loader],
                workflow=[('train', 500), ('val', -1)])
