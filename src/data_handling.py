"""
    Architecture:
    Run.py calls parse_dataset with the dataset name
    parse_dataset returns a list of Graph objects 
    Graph.py => Graph object
    Main.py => Arg parse
    data_handling => Load the data and return the list of Graph objects
                     Run calls this file again for train test split
    TODO: store the graphs in a pickle file and just load that instead of parsing the data everytime
"""
from typing import List

from config import DATA_PATH
import pandas as pd
import os.path
import networkx as nx
from graph import Graph
from sklearn.model_selection import StratifiedKFold
import numpy as np
import torch
import torch.nn.functional as F


def parse_dataset(name: str, degree_as_label: bool = False):
    dataset_folder_name = DATA_PATH + name + "/"
    edge_list_filename = dataset_folder_name + name + "_A.txt"
    graph_indicator_filename = (
        dataset_folder_name + name + "_graph_indicator.txt"
    )
    graph_label_filename = dataset_folder_name + name + "_graph_labels.txt"
    node_label_filename = dataset_folder_name + name + "_node_labels.txt"
    assert (
        os.path.isfile(edge_list_filename)
        and os.path.isfile(graph_indicator_filename)
        and os.path.isfile(graph_label_filename)
    ), "Dataset not found"

    # graph_id -> [nodes]
    # node -> graph_id
    node_list_df = pd.read_csv(graph_indicator_filename, header=None)
    node_list_df = node_list_df[0]
    node_to_graph_id = node_list_df.to_list()

    graph_count = len(set(node_to_graph_id))
    print(f"Found {graph_count} graphs in dataset")

    graph_labels_df = pd.read_csv(graph_label_filename, header=None)
    graph_labels = graph_labels_df[0].to_list()

    edge_list_df = pd.read_csv(edge_list_filename, header=None)
    edges = list(edge_list_df.itertuples(index=False, name=None))

    g_label_map = {}
    for label in graph_labels:
        if label not in g_label_map:
            g_label_map[label] = len(g_label_map)
    graphs: List[Graph] = [
        Graph(g_label_map[graph_labels[i]], nx.Graph(), i + 1, node_tags=[])
        for i in range(graph_count)
    ]
    # getting node labels
    if os.path.isfile(node_label_filename):
        print("Loading node labels from file")
        node_labels_df = pd.read_csv(node_label_filename, header=None)
        node_labels = node_labels_df[0].to_list()
    else:
        print("Setting node labels to zero")
        node_labels = [0] * len(node_to_graph_id)

    for idx, graph_id in enumerate(node_to_graph_id):
        graphs[graph_id - 1].g.add_node(idx + 1)

    for graph in graphs:
        graph.neighbours = [[] for _ in range(len(graph.g))]

    for node1, node2 in edges:
        graph_id = node_to_graph_id[node1 - 1]
        current_graph = graphs[graph_id - 1].g
        current_graph.add_edge(node1, node2)

    node_dicts = {}
    # Zero indexing the graphs
    for graph in graphs:
        node_list = list(graph.g.nodes())
        node_dict = {node_list[i]: i for i in range(len(node_list))}
        node_dicts[graph.id] = node_dict
        nx.relabel.relabel_nodes(graph.g, node_dict, copy=False)

        # setting the node tags to zero for filling
        graph.node_tags = [0] * len(graph.g)

        # set the value of maximum neighbours for a given node in the graph

        # create edge mat
        # edges = [list(pair) for pair in graph.g.edges()]
        # edges.extend([[i, j] for j, i in edges])
        # graph.edge_mat = torch.LongTensor(edges).transpose(0, 1)

        # overwrite node tags if the flag is true
        if degree_as_label:
            graph.node_tags = list(dict(graph.g.degree).values())

    for graph in graphs:
        graph.neighbours = [[] for _ in range(len(graph.g))]
        for i, j in graph.g.edges():
            graph.neighbours[i].append(j)
            graph.neighbours[j].append(i)

    # Setting labels to nodes and indexing.
    node_label_map = {}
    for idx, graph_id in enumerate(node_to_graph_id):
        if node_labels[idx] not in node_label_map:
            node_label_map[node_labels[idx]] = len(node_label_map)

        graphs[graph_id - 1].node_tags[
            node_dicts[graph_id][idx + 1]
        ] = node_label_map[node_labels[idx]]

    # the count of labels encountered in the entire dataset
    num_labels = len(node_label_map)

    for graph in graphs:
        graph.node_features = F.one_hot(
            torch.tensor(graph.node_tags), num_classes=num_labels
        )
        graph.max_neighbour = max([len(x) for x in graph.neighbours])

    print("Number of unique graph labels", len(set(graph_labels)))
    print("Number of unique node labels", len(set(node_labels)))
    print("Number of graphs", len(graphs))

    return graphs, len(set(graph_labels))


# Graphs => list of graph objects
# The number of folds to train on
# Return an [(test_graphs, train_graphs)]
# Seed
def k_fold_splitter(graphs: List[Graph], seed: int, fold_count: int):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    labels = [graph.label for graph in graphs]
    skf.get_n_splits(graphs, labels)
    graphs = np.array(graphs)

    graph_list_folded = []
    for train_idxs, test_idxs in skf.split(labels, labels):
        print(train_idxs)
        graph_list_folded.append((graphs[train_idxs], graphs[test_idxs]))

    print(len(graph_list_folded))
    return graph_list_folded[:fold_count]


def main():
    graphs, classes_count = parse_dataset("PROTEINS")
    graph_list = k_fold_splitter(graphs, 42, 10)
    for test, train in graph_list:
        print(test.shape, train.shape)


if __name__ == "__main__":
    main()
# parse_dataset("REDDIT-MULTI-5K")
