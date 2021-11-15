import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        """
        Class for a simple one layer classifier with softmax at the end for classification.

        Inputs:
            input_dim: Input dimension, In context of GNN, it will be the size
                       of the embeddings.
            output_dim: Number of output labels, In context of GNN, it will be
                        the number of output labels.
        """
        self.layers = nn.Sequential(
            nn.Linear(input_dim, output_dim), nn.Softmax(dim=output_dim)
        )

    def forward(self, x):
        return self.layers(x)
