import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from collections import OrderedDict
import itertools
class Pruner(nn.Module):
    def __init__(self, model):
        super(Pruner, self).__init__()
        self.model = model
        self.to_remove = {}
        self.next_layer = {}

        layer_names = list(model.pruning_layers._modules.keys())
        for i, layer_name in enumerate(layer_names):
            layer = model.pruning_layers._modules.get(layer_name)
            if(i < len(model.pruning_layers._modules)-1):
                layer.layer_name = layer_name
                self.next_layer[layer_name] = \
                    model.pruning_layers._modules.get(layer_names[i+1])
                layer.register_forward_hook(self.min_L1)
            i += 1

    def min_L1(self, layer, input, output):
        activs = np.abs(input[0].numpy())
        L1 = np.apply_over_axes(np.sum, activs, [0,2,3])
        L1 = L1.reshape(L1.shape[1])
        self.to_remove[layer.layer_name] = [np.argmin(L1)]

    def forward(self, x):
        return self.model(x)

    def prune(self):
        i = 0
        for layer_name, seq_layer in model.pruning_layers._modules.items():
            if(i < len(model.pruning_layers._modules)-1):
                nodes_to_remove = self.to_remove[layer_name]
                n_remove = len(nodes_to_remove)

                layer = seq_layer._modules['0']
                layer.out_channels -= n_remove

                # delete layer_index row in layer, and column in next layer
                np_weights = layer.weight.data.cpu().numpy()
                np_weights = np.delete(np_weights, nodes_to_remove, axis=0)
                layer.weight = Parameter(torch.from_numpy(np_weights))#.cuda())

                layer_weights = layer.bias.data.cpu().numpy()
                layer_weights = np.delete(layer_weights, nodes_to_remove)
                layer.bias = Parameter(torch.from_numpy(layer_weights))#.cuda())

                next_layer = self.next_layer[layer_name]._modules['0']
                next_layer.in_channels -= 1
                np_weights = next_layer.weight.data.cpu().numpy()
                np_weights = np.delete(np_weights, nodes_to_remove, axis=1)
                next_layer.weight = Parameter(torch.from_numpy(np_weights))#.cuda())
            i += 1

if __name__ == '__main__':
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.pruning_layers = nn.Sequential(OrderedDict([
                ("Layer1",
                    nn.Sequential(
                    nn.Conv2d(1, 10, 5, padding=2),
                    nn.MaxPool2d(2),
                    nn.ReLU())),
                ("Layer2",
                    nn.Sequential(nn.Conv2d(10, 20, 5, padding=2),
                    nn.MaxPool2d(2),
                    nn.ReLU())),
                ("Layer3",
                    nn.Sequential(nn.Conv2d(20, 20, 5, padding=2),
                    nn.MaxPool2d(2),
                    nn.ReLU())),
                ("Layer4",
                    nn.Sequential(nn.Conv2d(20, 20, 5, padding=2),
                    nn.MaxPool2d(2),
                    nn.ReLU()))
                ]))
            self.fc1 = nn.Linear(20, 10)

        def forward(self, x):
            print(x.shape)
            x = self.pruning_layers(x)
            print(x.shape)
            x = x.view(-1, 20)
            print(x.shape)
            x = self.fc1(x)
            print(x.shape)
            return F.log_softmax(x, dim=1)

    model = Net()

    pruning_model = Pruner(model)

    def train(model, device, train_loader, optimizer, epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

    def test(model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
                pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

        return 1. * correct / len(test_loader.dataset)

    torch.manual_seed(1)

    device = torch.device("cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=64, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=64, shuffle=True, **kwargs)

    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    for epoch in range(1):
        #train(model, device, train_loader, optimizer, epoch)
        #test(model, device, test_loader)
        pass
    val_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=10000, shuffle=True, **kwargs)
    test(pruning_model, device, val_loader)

    pruning_model.prune()
