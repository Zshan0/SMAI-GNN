from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sys

from .gnnlayer import GNNLayer
from .mlp import MLP

sys.path.append("..")
from graph import *


class GINLayer(GNNLayer):
    def __init__(self, input_dim: int, output_dim: int, num_mlp_layers: int, use_weights: bool = True,
                 hidden_dim: int = 5):
        """
        Class for a general layer of a graph neural network that is based on the
        Graph class defined. Serves as a base class for different types of GNN
        layers.

        Inputs:
            input_dim: Size of input features of the nodes.
            output_dim: Final output size of the embedding.
            use_weights: Bool to indicidate if the weights are being used.
        """
        super().__init__(input_dim, output_dim, use_weights)
        self.num_mlp_layers = num_mlp_layers
        self.mlp = MLP(input_dim, [hidden_dim] * num_mlp_layers, output_dim)
        self.eps = nn.Parameter(torch.zeros(1))
        # self.input_dim = input_dim
        # self.output_dim = output_dim
        #
        # if use_weights:
        #     self.W = nn.Parameter(torch.FloatTensor(output_dim, input_dim))
        #
        # # weight initialization
        # for param in self.parameters():
        #     nn.init.xavier_uniform_(param)

    def aggregate(self, H, combined_neighbours):
        """
        Function that is overloaded by children classes. The base class
        will use graphSAGE aggergate function which is:
            MAX(ReLU(W \times h_u^{k - 1}), \forall u N(v))

        Inputs:
            H: 2D Tensor of all feature vectors. [i, j] represents jth feature
               of  ith node
            combined_neighbours: 2D Tensor of neighbours of all nodes.
                                 [i, j] is the jth neighbour of ith node.
        """
        zeros = torch.zeros((1, H.shape[0]))
        new_H = torch.cat([H, zeros.reshape((1, -1))])
        new_H = new_H[combined_neighbours]
        new_H = torch.sum(new_H, axis=1)
        new_H = new_H + (1 + self.eps) * H
        new_H = self.mlp(new_H.t)
        new_H = F.relu(new_H)

        return new_H

    def combine(self, H, a):
        # useless
        """
        Function that is overloaded by children classes. The base class
        will not have any combine function.

        Inputs:
            H: 3D Tensor of feature vector of neighbours. [i, j, k] represents
               the kth feature of jth neighbour of ith node.
            a: 2D Tensor of all feature vectors. [i, j] represents jth feature
               of  ith node
        """
        return H

    def forward(self, H, combined_neighbours):
        """
        Common function which applies combine and aggregate in sequence. Need
        not be overloaded.

        Inputs:
            node_features: row-vector of size 1 x input_dim.
            neigh_features: 2D Tensor of size num_neighbors x input_dim

        Returns: [num_neighbors x output_dim] node embedding
        """
        return self.aggregate(H, combined_neighbours)
