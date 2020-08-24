'''
Neural network for MNIST
'''

from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    '''
    Neural network for MNIST
    '''
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.pruning_layers = nn.Sequential(OrderedDict([
            ("Layer1",
             nn.Sequential(
                 nn.Conv2d(1, 10, 5, padding=2),
                 nn.MaxPool2d(2),
                 nn.ReLU(), nn.BatchNorm2d(10))),
            ("Layer2",
             nn.Sequential(
                 nn.Conv2d(10, 20, 5, padding=2),
                 nn.MaxPool2d(2),
                 nn.ReLU(), nn.BatchNorm2d(20))),
            ("Layer3",
             nn.Sequential(
                 nn.Conv2d(20, 20, 5, padding=2),
                 nn.MaxPool2d(2),
                 nn.ReLU(), nn.BatchNorm2d(20))),
            ("Layer4",
             nn.Sequential(
                 nn.Conv2d(20, 20, 5, padding=2),
                 nn.MaxPool2d(2),
                 nn.ReLU(), nn.BatchNorm2d(20)))
        ]))
        self.fc_layers = nn.Linear(20, 10)

    def forward(self, x):
        x = self.pruning_layers(x)
        x = x.view(-1, 20)
        x = self.fc_layers(x)
        return F.log_softmax(x, dim=1)
