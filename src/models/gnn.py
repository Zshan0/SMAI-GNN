from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .gnnlayer import GNNLayer
from .classifier import Classifier

import sys

sys.path.append("..")
from graph import *


class GNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_layers: int,
        hidden_dim: int,
        output_dim: int,
        is_concat: bool = False,
    ):
        """
        General class for a graph neural network that is based on the Graph
        class defined. Serves as a base class for different types of GNNs.

        Inputs:
            input_dim: Size of input features of the nodes.
            hidden_dim: Size of hidden features of the nodes.
            num_layers: Number of GNN layers to be considered.
            output_dim: Total number of classes for graphs.
            is_concat: If the combine function involves concatenation.
        """
        super(GNN, self).__init__()

        self.input_dim = input_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.is_concat = is_concat

        self.layers = nn.ModuleList()
        self.layers.append(GNNLayer(input_dim, hidden_dim))
        # The output of ith layer will be passed to (i + 1)th layer
        for i in range(1, num_layers):
            self.layers.append(GNNLayer(hidden_dim, hidden_dim))

        self.classifiers = nn.ModuleList()
        self.classifiers.append(Classifier(input_dim, output_dim))
        for i in range(1, num_layers):
            self.classifiers.append(Classifier(hidden_dim, output_dim))

    def combine_batch(self, graph_batch):
        """
        Sets the index for all the nodes together so that the functions
        can be run together faster.

        Inputs:
            graph_batch: list of graphs
        """
        max_degree = max([graph.max_neighbour for graph in graph_batch])
        # print("MD", max_degree)
        if not self.is_concat:
            # self loops may not be considered for max degree
            max_degree += 1

        total_nodes = sum([len(graph.g) for graph in graph_batch])

        cur_num = 0

        neighbour_list = np.full((total_nodes, max_degree), -1)
        for _, graph in enumerate(graph_batch):
            nodes = len(graph.g)
            for ind, neighbours in enumerate(graph.neighbours):
                if not self.is_concat:
                    neighbour_list[cur_num + ind][0 : len(neighbours) + 1] = (
                        np.array(neighbours + [ind]) + cur_num
                    )
                else:
                    neighbour_list[cur_num + ind][0 : len(neighbours)] = (
                        np.array(neighbours) + cur_num
                    )

            cur_num += nodes

        return torch.Tensor(neighbour_list).long()

    def readout(self, H, graph_cumulative):
        """
        Creates embedding for the graph by summing all the embeds of the graph.

        Inputs:
            H: 2D matrix of node features ordered based on graphs.
            graph_cumulative: node index range of each graph
        """
        graph_embed = []
        for ind in range(len(graph_cumulative) - 1):
            graph_embed.append(
                torch.sum(
                    H[graph_cumulative[ind] : graph_cumulative[ind + 1]], axis=0
                )
            )

        return torch.stack(graph_embed)

    def classify(self, graph_embeds):
        """
        Takes a list of graph embeddings and runs a simple linear
        classifier on it.

        Inputs:
            graph_embeds: List of graph embeds,
            dimensions: [num_graphs x node_features]
        """
        predictions = []
        for ind, x in enumerate(graph_embeds):
            predictions.append(self.classifiers[ind](x))
        predictions = torch.stack(predictions)
        # sum across all the intermediate predictions
        return torch.sum(predictions, axis=0)

    def forward(self, graph_batch: List[Graph]):
        """
        Common function that applies each layer sequentially and stores all the
        embeddings.

        Inputs:
            graph_batch: list of graphs
        """
        # dim(graph_features) = total_nodes x dim(node_features)
        graph_features = []
        for graph in graph_batch:
            graph_features.extend(graph.node_features)

        graph_features = torch.stack(graph_features).float()
        graph_cumulative = np.cumsum(
            [0] + [len(graph.g) for graph in graph_batch]
        )

        combined_neighbours = self.combine_batch(graph_batch)

        H = graph_features
        graph_embeds = [self.readout(H, graph_cumulative)]
        for layer in range(self.num_layers - 1):
            H = self.layers[layer](H, combined_neighbours)
            graph_embeds.append(self.readout(H, graph_cumulative))

        return self.classify(graph_embeds)
