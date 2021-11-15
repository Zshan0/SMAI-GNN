from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: List, output_dim: int):
        """
        Class for Multi Layer Perceptron to be used in GINs. The MLP can be a
        single linear layer too. Uses Batch Normalization for every hidden
        layer.

        Input:
            input_dim: Dimensions of input
            hidden_dim: Dimensions of hidden layers
        """
        super(MLP, self).__init__()

        self.num_layers = len(hidden_dim) + 2  # hidden + input + output

        self.layers = nn.ModuleList()
        self.layers.append(nn.Flatten())
        layer_dim = [input_dim] + hidden_dim + [output_dim]

        for ind in range(len(layer_dim) - 1):
            if (
                ind > 0
            ):  # Activation and batch normalization in all except the first layer
                self.layers.append(nn.BatchNorm1d((layer_dim[ind])))
                self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(layer_dim[ind], layer_dim[ind + 1]))

        self.sequence = nn.Sequential(*self.layers)

    def forward(self, x):
        '''Forward pass'''
        return self.sequence(x)
