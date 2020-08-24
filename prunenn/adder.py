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
Adds nodes to neural network
'''
class Adder(nn.Module):
  def __init__(self, model, thres = 0.4, function= "var"):
        super(Adder, self).__init__()
        self.model = model
        self.to_add = {}
        self.next_layer = {}
        self.thres = thres
        self.to_train = True #true is train, false is test

        layer_names = list(model.pruning_layers._modules.keys())
        for i, layer_name in enumerate(layer_names):
            layer = model.pruning_layers._modules.get(layer_name)
            if(i < len(model.pruning_layers._modules)-1):
                layer.layer_name = layer_name
                self.next_layer[layer_name] = \
                    model.pruning_layers._modules.get(layer_names[i+1])
                if(function == "var" ): #and not to_train
                    layer.register_forward_hook(self.low_variance_nodes)
            i += 1

    def add(self):
        i = 0
        for layer_name, seq_layer in model.pruning_layers._modules.items():
            if(i < len(model.pruning_layers._modules) - 1):
                layer = seq_layer._modules['0']

                nodes_to_add = self.to_add[layer_name]
                np_weights = layer.weight.data.cpu().numpy()

                original_weights_to_add = np_weights[nodes_to_add, :, :, :]
                np_weights[nodes_to_add, :, :, :] = np.random.normal(original_weights_to_add, 0.5 * np.std(original_weights_to_add))
                weights_to_add = np_weights[nodes_to_add, :, :, :]
                weights_to_add = np.random.normal(weights_to_add, 0.5 * np.std(weights_to_add))

                np_weights = np.concatenate((np_weights, weights_to_add), axis = 0).astype(np.float)
                np_weights = np.random.normal(np_weights,0.5 * np.std(np_weights))
                layer.weight = Parameter(torch.from_numpy(np_weights).type(torch.FloatTensor).cuda())

                num_nodes_to_add = len(nodes_to_add)
                # increment out features
                layer.out_channels += num_nodes_to_add

                # increment in features of next layer
                next_layer = self.next_layer[layer_name]
                next_layer = next_layer._modules['0']
                next_layer.in_channels += num_nodes_to_add

                bias_weights = layer.bias.data.cpu().numpy()
                bias_weights[nodes_to_add] = np.random.normal(bias_weights[nodes_to_add], 0.5 * np.std(bias_weights[nodes_to_add]))
                bias_to_add = bias_weights[nodes_to_add]
                bias_to_add = np.random.normal(bias_to_add,0.5*np.std(bias_to_add))

                bias_weights = np.concatenate((bias_weights, bias_to_add)).astype(np.float)
                bias_weights = np.random.normal(bias_weights,0.5*np.std(bias_weights))
                layer.bias = Parameter(torch.from_numpy(bias_weights).type(torch.FloatTensor).cuda())

                # add dimension to kernels
                np_weights = next_layer.weight.data.cpu().numpy()
                np_weights = np.concatenate((np_weights, np_weights[:, nodes_to_add, :, :]), axis = 1)
                next_layer.weight = Parameter(torch.from_numpy(np_weights).cuda())

                # update batch layer
                batch_layer = seq_layer._modules['3']

                running_mean = batch_layer.running_mean.data.cpu().numpy()
                running_mean = np.concatenate((running_mean, running_mean[nodes_to_add]))
                batch_layer.running_mean = torch.from_numpy(running_mean).cuda()

                batch_weight = batch_layer.weight.data.cpu().numpy()
                batch_weight = np.concatenate((batch_weight, batch_weight[nodes_to_add]))
                batch_layer.weight = Parameter(torch.from_numpy(batch_weight).cuda())

                batch_bias = batch_layer.bias.data.cpu().numpy()
                batch_bias = np.concatenate((batch_bias, batch_bias[nodes_to_add]))
                batch_layer.bias = Parameter(torch.from_numpy(batch_bias).cuda())

                running_var = batch_layer.running_var.data.cpu().numpy()
                running_var = np.concatenate((running_var, running_var[nodes_to_add]))
                batch_layer.running_var = torch.from_numpy(running_var).cuda()
        i+=1

  def forward(self, x):
        return self.model(x)

  def low_variance_nodes(self, layer, input, output):
      if(not self.to_train):
        h, w = output.shape[2], output.shape[3]
        # get correlations
        layer_vars = np.apply_over_axes(np.var, output.cpu().detach().numpy(), [0,2,3])
        layer_vars = np.abs(layer_vars)
        self.to_add[layer.layer_name] = np.where((layer_vars.ravel() < self.thres))[0]
