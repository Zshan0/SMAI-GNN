from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sys

sys.path.append("..")
from graph import *


class GNNLayer(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, use_weights: bool = True
    ):
        """
        Class for a general layer of a graph neural network that is based on the
        Graph class defined. Serves as a base class for different types of GNN
        layers.

        Inputs:
            input_dim: Size of input features of the nodes.
            output_dim: Final output size of the embedding.
            use_weights: Bool to indicidate if the weights are being used.
        """
        super(GNNLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        if use_weights:
            self.W = nn.Parameter(torch.FloatTensor(output_dim, input_dim))

        # weight initialization
        for param in self.parameters():
            nn.init.xavier_uniform_(param)

    def aggregate(self, H, combined_neighbours):
        """
        Function that is overloaded by children classes. The base class
        will use graphSAGE aggergate function which is:
            MAX(ReLU(W \times h_u^{k - 1}), \forall u N(v))

        Inputs:
            H: 2D Matrix of all feature vectors. [i, j] represents jth feature
               of  ith node
            combined_neighbours: 2D matrix of neighbours of all nodes.
                                 [i, j] is the jth neighbour of ith node.
        """
        new_H = F.relu(self.W.mm(H.t())).t()

        # min of all nodes will serve as invariant for max
        min_row = torch.min(new_h, dim=0).values
        new_H = torch.cat([new_H, min_row.t()])

        # combine after considering the invariant into the picture.
        new_H = self.combine(new_H[combined_neighbours], H)
        return torch.max(new_H, dim=1).values

    def combine(self, H, a):
        """
        Function that is overloaded by children classes. The base class
        will not have any combine function.

        Inputs:
            H: 3D matrix of feature vector of neighbours. [i, j, k] represents
               the kth feature of jth neighbour of ith node.
            a: 2D matrix of all feature vectors. [i, j] represents jth feature
               of  ith node
        """
        return H

    def forward(self, H, combined_neighbours):
        """
        Common function which applies combine and aggregate in sequence. Need
        not be overloaded.

        Inputs:
            node_features: row-vector of size 1 x input_dim.
            neigh_features: 2D matrix of size num_neighbors x input_dim

        Returns: [num_neighbors x output_dim] node embedding
        """
        return self.aggregate(H, combined_neighbours)
