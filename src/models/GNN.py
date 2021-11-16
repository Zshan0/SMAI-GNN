from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from GNNLayer import GNNLayer
from Classifier import Classifier

import sys

sys.path.append("..")
from graph import *


class GNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_layers: int,
        output_dim: int,
        graph_class_num: int,
        is_concat: bool = False,
    ):
        """
        General class for a graph neural network that is based on the Graph
        class defined. Serves as a base class for different types of GNNs.

        Inputs:
            input_dim: Size of input features of the nodes.
            num_layers: Number of GNN layers to be considered.
            output_dim: Final output size of the embedding.
            graph_class_num: Total number of classes for graphs.
            is_concat: If the combine function involves concatenation.
        """
        super(GNN, self).__init__()

        self.input_dim = input_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.is_concat = is_concat
        self.graph_class_num = graph_class_num

        self.layers = nn.ModuleList()
        self.layers.append(GNNLayer(input_dim, output_dim, is_concat))
        # The output of ith layer will be passed to (i + 1)th layer
        for i in range(1, num_layers):
            self.layers.append(GNNLayer(output_dim, output_dim, is_concat))

        self.classifier = Classifier(input_dim, graph_class_num)

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
                np.sum(H[graph_cumulative[ind] : graph_cumulative[ind + 1]])
            )
        return torch.Tensor(graph_embed)

    def classify(self, graph_embeds):
        """
        Takes a list of graph embeddings and runs a simple linear
        classifier on it.

        Inputs:
            graph_embeds: List of graph embeds,
            dimensions: [num_graphs x node_features]
        """
        predictions = []
        for x in graph_embeds:
            predictions.append(self.classifier(x))
        return predictions

    def forward(self, graph_batch: List[Graph]):
        """
        Common function that applies each layer sequentially and stores all the
        embeddings.

        Inputs:
            graph_batch: list of graphs
        """
        # dim(graph_features) = total_nodes x dim(node_features)
        graph_features = torch.cat(
            [graph.node_features for graph in graph_batch], 0
        )
        graph_cumulative = np.cumsum(
            [0] + [len(graph) for graph in graph_batch]
        )

        combined_neighbours = self.combine_batch(graph_batch)

        graph_embeds = []
        H = graph_features
        for layer in range(self.num_layers - 1):
            graph_embeds.append(self.readout(H, graph_cumulative))
            H = self.layers[layer](H, combined_neighbours)

        return self.classify(graph_embeds)
