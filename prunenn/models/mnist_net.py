import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

class MNIST_Net(nn.Module):
    def __init__(self):
        super(MNIST_Net, self).__init__()
        self.pruning_layers = nn.Sequential(OrderedDict([
            ("Layer1",
                nn.Sequential(
                nn.Conv2d(1, 10, 5, padding=2),
                nn.MaxPool2d(2),
                nn.ReLU(), nn.BatchNorm2d(10))),
            ("Layer2",
                nn.Sequential(nn.Conv2d(10, 20, 5, padding=2),
                nn.MaxPool2d(2),
                nn.ReLU(), nn.BatchNorm2d(20))),
            ("Layer3",
                nn.Sequential(nn.Conv2d(20, 20, 5, padding=2),
                nn.MaxPool2d(2),
                nn.ReLU(), nn.BatchNorm2d(20))),
            ("Layer4",
                nn.Sequential(nn.Conv2d(20, 20, 5, padding=2),
                nn.MaxPool2d(2),
                nn.ReLU(), nn.BatchNorm2d(20)))
            ]))
        self.fc1 = nn.Linear(20, 10)

    def forward(self, x):

        x = self.pruning_layers(x)
        x = x.view(-1, 20)
        x = self.fc1(x)

        return F.log_softmax(x, dim=1)
