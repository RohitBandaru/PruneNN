'''
Neural network for CIFAR
'''

from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F

class CIFARNet(nn.Module):
    '''
    Neural network for CIFAR
    '''
    def __init__(self):
        super(CIFARNet, self).__init__()
        self.pruning_layers = nn.Sequential(OrderedDict([
            ("Layer1",
             nn.Sequential(
                 nn.Conv2d(3, 64, 5, padding=2),
                 nn.MaxPool2d(2),
                 nn.ReLU(),
                 nn.BatchNorm2d(64, track_running_stats=True)
             )),
            ("Layer2",
             nn.Sequential(
                 nn.Conv2d(64, 128, 5, padding=2),
                 nn.MaxPool2d(2),
                 nn.ReLU(),
                 nn.Dropout(0.5)
             )),
            ("Layer3",
             nn.Sequential(
                 nn.Conv2d(128, 256, 5, padding=2),
                 nn.ReLU(),
                 nn.BatchNorm2d(256, track_running_stats=True)
             )),
            ("Layer4",
             nn.Sequential(
                 nn.Conv2d(256, 256, 5, padding=2),
                 nn.MaxPool2d(2),
                 nn.ReLU(),
                 nn.Dropout(0.5)
             )),
            ("Layer5",
             nn.Sequential(
                 nn.Conv2d(256, 512, 5, padding=2),
                 nn.ReLU(),
                 nn.BatchNorm2d(512, track_running_stats=True),
             )),
            ("Layer6",
             nn.Sequential(
                 nn.Conv2d(512, 512, 5, padding=2),
                 nn.MaxPool2d(2),
                 nn.ReLU(),
                 nn.BatchNorm2d(512, track_running_stats=True),
                 nn.Dropout(0.5)
             )),
            ("Layer7",
             nn.Sequential(
                 nn.Conv2d(512, 512, 5, padding=2),
                 nn.ReLU(),

             )),
            ("Layer8",
             nn.Sequential(
                 nn.Conv2d(512, 512, 5, padding=2),
                 nn.MaxPool2d(2),
                 nn.ReLU(),
                 nn.BatchNorm2d(512, track_running_stats=True),
                 nn.Dropout(0.5)
             ))
        ]))
        self.fc_layers = nn.Sequential(
            nn.Linear(512, 512),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.pruning_layers(x)
        x = x.view(-1, 512)
        x = self.fc_layers(x)
        return F.log_softmax(x, dim=1)
