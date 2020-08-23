import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from collections import OrderedDict
import itertools

'''
Remove nodes from neural network
'''
class Pruner(nn.Module):
    def __init__(self, model, thres, function="corrs"):
        super(Pruner, self).__init__()
        self.model = model
        self.to_remove = {}
        self.next_layer = {}
        self.thres = thres

        layer_names = list(model.pruning_layers._modules.keys())
        for i, layer_name in enumerate(layer_names):
            layer = model.pruning_layers._modules.get(layer_name)
            if(i < len(model.pruning_layers._modules)-1):
                layer.layer_name = layer_name
                self.next_layer[layer_name] = \
                    model.pruning_layers._modules.get(layer_names[i+1])
                if(function == "corrs"):
                    layer.register_forward_hook(self.correlations)
                elif(function == "l1"):
                    layer.register_forward_hook(self.min_L1)
            i += 1

    def prune(self):
        i = 0
        for layer_name, seq_layer in self.model.pruning_layers._modules.items():
            if(i < len(self.model.pruning_layers._modules)-1):
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

                batch_layer = seq_layer._modules['3']

                running_mean = batch_layer.running_mean.data.cpu().numpy()
                running_mean = np.delete(running_mean, nodes_to_remove, axis=0)
                batch_layer.running_mean = torch.from_numpy(running_mean)#.cuda()

                batch_weight = batch_layer.weight.data.cpu().numpy()
                batch_weight = np.delete(batch_weight, nodes_to_remove)
                batch_layer.weight = Parameter(torch.from_numpy(batch_weight))#.cuda())

                batch_bias = batch_layer.bias.data.cpu().numpy()
                batch_bias = np.delete(batch_bias, nodes_to_remove)
                batch_layer.bias = Parameter(torch.from_numpy(batch_bias))#.cuda())

                running_var = batch_layer.running_var.data.cpu().numpy()
                running_var = np.delete(running_var, nodes_to_remove)
                batch_layer.running_var = torch.from_numpy(running_var)#.cuda()
            i += 1

    def forward(self, x):
        return self.model(x)
    '''
    PRUNING FUNCTIONS
    START HERE
    '''
    def min_L1(self, layer, input, output):
        activs = np.abs(input[0].numpy())
        L1 = np.apply_over_axes(np.sum, activs, [0,2,3])
        L1 = L1.reshape(L1.shape[1])
        self.to_remove[layer.layer_name] = [np.argmin(L1)]

    def correlations(self, layer, input, output):
        h, w = output.shape[2], output.shape[3]
        # get correlations
        n_filters = output.shape[1]
        corrs = np.zeros((n_filters, n_filters))
        for i in range(h):
            for j in range(w):
                ap = output[:,:,i,j]
                corrs += np.corrcoef(ap.detach().numpy().T)
        corrs /= n_filters

        # find filter pairs above correlation threshold
        corrs = np.abs(corrs)
        np.fill_diagonal(corrs, 0)
        sorted_corr = np.sort(corrs.ravel())

        nodes_to_prune = np.where((self.thres <= corrs[:,:]))
        rows, cols = nodes_to_prune[0], nodes_to_prune[1]

        # get filters to remove
        for r in range(0, len(rows)):
            r_i = rows[r]
            if(r_i > -1):
              ind = np.where(cols[r] == rows[:])
              rows[ind] = -1
              ind = np.where(rows[r] == rows[r+1:])
              rows[ind] = -1
        self.to_remove[layer.layer_name] = rows[np.where(rows[:] > -1)]
