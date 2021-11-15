from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sys

sys.path.append("..")
from graph import *


class GNNLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, is_concat: bool):
        """
        Class for a general layer of a graph neural network that is based on the
        Graph class defined. Serves as a base class for different types of GNN
        layers.

        Inputs:
            input_dim: Size of input features of the nodes.
            output_dim: Final output size of the embedding.
            is_concat: If the combine function involves concatenation.
        """
        super(GNNLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_concat = is_concat

        if is_concat:
            self.W = nn.Parameter(torch.FloatTensor(2 * output_dim, input_dim))
        else:
            self.W = nn.Parameter(torch.FloatTensor(output_dim, input_dim))

        # weight initialization
        for param in self.parameters():
            nn.init.xavier_uniform_(param)

    def aggregate(self, H):
        """
        Function that is overloaded by children classes. The base class
        will use graphSAGE aggergate function which is:
            MAX(ReLU(W \times h_u^{k - 1}), \forall u N(v))

        Inputs:
            H: row-wise 2D matrix of neighbouring features after COMBINE.
        """
        combined = F.relu(self.W.mm(H.t())).t()
        # taking max of each column i.e feature
        return torch.max(combined, dim=0).values

    def combine(self, H, a):
        """
        Function that is overloaded by children classes. The base class
        will use graphSAGE combine function which is:
            h = [h, a]

        Inputs:
            H: row-wise 2D matrix of neighbouring features after COMBINE.
            a: node feature row-vector.
        """
        if self.is_concat:
            return torch.cat([H, a], 1)
        else:
            return H

    def forward(self, node_features, neigh_features):
        """
        Common function which applies combine and aggregate in sequence. Need
        not be overloaded.

        Inputs:
            node_features: row-vector of size 1 x input_dim.
            neigh_features: 2D matrix of size num_neighbors x input_dim

        Returns: [num_neighbors x output_dim] node embedding
        """
        return self.aggregate(self.combine(neigh_features, node_features))
