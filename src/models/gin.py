from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .GINLayer import GINLayer
from .classifier import Classifier
from .gnn import GNN

import sys

sys.path.append("..")
from graph import *


class GIN(GNN):
    def __init__(
        self,
        input_dim: int,
        num_layers: int,
        hidden_dim: int,
        num_mlp_layers: int,
        hidden_mlp_dim: int,
        output_dim: int,
        is_concat: bool = False,
    ):
        """
        GIN class which uses GINLayer instead of GNN.
        Inherited from GNN for all the functions

        Inputs:
            input_dim: Size of input features of the nodes.
            num_layers: Number of GNN layers to be considered.
            hidden_dim: Size of hidden features of the nodes.
            num_mlp_layers: Number of layers MLP used in aggregate.
            hidden_mlp_dim: Size of layers MLP used in aggregate.
            output_dim: Total number of classes for graphs.
            is_concat: If the combine function involves concatenation.
        """
        # False at the end because it is no longer using base class GNN layers
        super(GIN, self).__init__(
            input_dim, num_layers, hidden_dim, output_dim, False, False
        )

        self.layers = nn.ModuleList()
        # total number of layers required are self.layers - 1
        self.layers.append(
            GINLayer(
                input_dim, hidden_dim, num_mlp_layers, False, hidden_mlp_dim
            )
        )
        for _ in range(2, num_layers):
            self.layers.append(
                GINLayer(
                    hidden_dim,
                    hidden_dim,
                    num_mlp_layers,
                    False,
                    hidden_mlp_dim,
                )
            )

        self.classifiers = nn.ModuleList()
        self.classifiers.append(Classifier(input_dim, output_dim))
        for _ in range(1, num_layers):
            self.classifiers.append(Classifier(hidden_dim, output_dim))
