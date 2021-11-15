from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from GNNLayer import GNNLayer

import sys

sys.path.append("..")
from graph import *


class GNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_layers: int,
        output_dim: int,
        is_concat: bool = False,
    ):
        """
        General class for a graph neural network that is based on the Graph
        class defined. Serves as a base class for different types of GNNs.

        Inputs:
            input_dim: Size of input features of the nodes.
            num_layers: Number of GNN layers to be considered.
            output_dim: Final output size of the embedding.
            is_concat: If the combine function involves concatenation.
        """
        super(GNN, self).__init__()

        self.input_dim = input_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.is_concat = is_concat

        self.layers = nn.ModuleList()
        self.layers.append(GNNLayer(input_dim, output_dim, is_concat))
        # The output of ith layer will be passed to (i + 1)th layer
        for i in range(1, num_layers):
            self.layers.append(GNNLayer(output_dim, output_dim, is_concat))

    def combine_batch(self, graph_batch):
        """
        Sets the index for all the nodes together so that the functions
        can be run together faster.

        Inputs:
            graph_batch: list of graphs
        """
        max_degree = max([graph.max_degree for graph in graph_batch])

        total_nodes = sum([len(graph) for graph in graph_batch])

        cur_num = 0

        neighbour_list = np.full((total_nodes, max_degree), -1)
        for graph_num, graph in enumerate(graph_batch):
            nodes = len(graph)
            for ind, neighbours in enumerate(graph.neighbours):
                if not self.concat:
                    neighbour_list[cur_num + ind][
                        0 : len(neighbours) + 1
                    ] = neighbours + [ind]
                else:
                    neighbour_list[cur_num + ind][
                        0 : len(neighbours)
                    ] = neighbours

            cur_num += nodes

        return torch.Tensor(neighbour_list)

    def forward(self, graph_batch: List[Graph]):
        """
        Common function that applies each layer sequentially and stores all the
        embeddings.

        Inputs:
            graph_batch: list of graphs
        """
        # dim(graph_features) = batch x num_neighbour x dim(node_features)
        graph_features = torch.cat(
            [graph.node_features for graph in graph_batch], 0
        )

        combined_neighbours = self.combine_batch(graph_batch)

        hidden_features = [graph_features]
        h = graph_features
        for layer in range(self.num_layers - 1):
            h = self.layers[layer](h)
