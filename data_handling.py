"""
    Architecture:
    Run.py calls parse_dataset with the dataset name
    parse_dataset returns a list of Graph objects 
    Graph.py => Graph object
    Main.py => Arg parse
    data_handling => Load the data and return the list of Graph objects
                     Run calls this file again for train test split
"""
from config import DATA_PATH
import pandas as pd
import os.path
import networkx as nx
from graph import Graph


def parse_dataset(name: str):
    dataset_folder_name = DATA_PATH + name + "/"
    edge_list_filename = dataset_folder_name + name + "_A.txt"
    graph_indicator_filename = (
            dataset_folder_name + name + "_graph_indicator.txt"
    )
    graph_label_filename = dataset_folder_name + name + "_graph_labels.txt"
    #     print(edge_list_filename, graph_indicator_filename, graph_label_filename)
    node_label_filename = dataset_folder_name + name + "_node_labels.txt"
    assert (
            os.path.isfile(edge_list_filename)
            and os.path.isfile(graph_indicator_filename)
            and os.path.isfile(graph_label_filename)
    ), "Dataset not found"

    if os.path.isfile(node_label_filename):
        print("Loading node labels from file")
    else:
        print("Setting node labels to zero")

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
    print(edge_list_df)

    edges = list(edge_list_df.itertuples(index=False, name=None))
    graphs = [Graph(graph_labels[i], nx.Graph(), i + 1) for i in range(graph_count)]

    for idx, node in enumerate(node_to_graph_id):
        graphs[node - 1].g.add_node(idx + 1)

    for node1, node2 in edges:
        node1 = int(node1)
        node2 = int(node2)
        graph_id = node_to_graph_id[node1 - 1]
        current_graph = graphs[graph_id - 1].g
        current_graph.add_edge(node1, node2)

    return graphs


def main():
    parse_dataset("PROTEINS")


if __name__ == "__main__":
    main()
