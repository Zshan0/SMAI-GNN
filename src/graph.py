from typing import Optional, List

import numpy as np


class Graph:
    def __init__(self, label=None, g=None, id=None, node_tags=None):
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors: List[List[int]] = []
        self.node_features: Optional[np.array] = None
        self.max_neighbor = 0
        self.id = id

    def __str__(self):
        return f"Graph {self.id} with label {self.label} and {len(self.g)} nodes and tags are {self.node_tags}"

    def __copy__(self):
        pass

def main():
    pass


if __name__ == "__main__":
    main()
