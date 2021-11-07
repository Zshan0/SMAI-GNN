class Graph:
    def __init__(self, label, g, id, node_tags=None):
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0
        self.max_neighbor = 0
        self.id = id

    def __str__(self):
        return f"Graph {self.id} with label {self.label} and {len(self.g)} nodes and tags are {self.node_tags}"


def main():
    pass


if __name__ == "__main__":
    main()
