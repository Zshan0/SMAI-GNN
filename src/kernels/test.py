import sys

from networkx import classes
from graph_analysis import *
from sklearn.model_selection import train_test_split
import copy
import random
from WLSubtree_Kernel import WL
import networkx as nx
from matplotlib import pyplot as plt

try:
    from graph import Graph
    from data_handling import *
    from config import DATA_PATH_WL
except:
    sys.path.append("..")
    from graph import Graph
    from data_handling import *
    from config import DATA_PATH_WL


def generate_adjacency_list(G):
    adj = [0] * len(G)
    for s, nbrs in G.adjacency():
        adj[s] = []
        for nbr in nbrs.keys():
            adj[s].append(nbr)
    return adj


def accuracy(testing_graphs, training_graphs, avg_nodes_0, avg_nodes_1):
    """
    Returns the accuracy for the testing graphs provided
    simple metric for accuracy = no of correct answer/ no of questions asked
    Now how do we know if the answer is correct or not
    We will find the average similarity of it with label 0 and the average
    similarity of it with label 1 and whichever one is the highest assign it
    that label. Correct if it matches with the actual label else wrong.
    """
    n = len(testing_graphs)
    m = len(training_graphs)
    accuracy = 0
    for test_graph in testing_graphs:
        average_0 = 0.0
        average_1 = 0.0
        cnt_0 = 1
        cnt_1 = 1
        answer = 1
        for train_graph in training_graphs:
            G = copy.deepcopy(test_graph)
            H = copy.deepcopy(train_graph)
            sim = WL(G, H, num_iter=7).train()
            if train_graph.label == 0:
                average_0 += sim
                cnt_0 += 1
            else:
                average_1 += sim
                cnt_1 += 1
        # print(average_0, average_1, len(test_graph.g))
        average_0 /= (cnt_0 * avg_nodes_0 * len(test_graph.g))
        average_1 /= (cnt_1 * avg_nodes_1 * len(test_graph.g))
        # print(f"Average sim of {test_graph.label} with:")
        # print(
        #     f"label 0 graphs = {average_0}, and label 1 graphs = {average_1}")
        if average_0 > average_1:
            answer = 0
        accuracy += (answer == test_graph.label)
    return accuracy / n


def main(num):
    """
    Average similarity for similar graphs and
    average similarity for dissimilar graphs
    """
    graphs, classes_count = parse_dataset("PROTEINS", False, DATA_PATH_WL)
    graphs_train, graphs_test = train_test_split(
        graphs, shuffle=True, test_size=0.33)
    graphs = graphs_train[:num]
    graphs_test = graphs_test[:10]

    """
    Use graph analysis as defined in 'graph_analysis.py'
    """
    g = copy.deepcopy(graphs)
    x1, x2, y1, y2 = analyse(g, classes_count)

    a00 = a01 = a11 = 0
    count_00 = count_01 = count_11 = 0
    for i in range(len(graphs)):
        for j in range(i + 1, len(graphs)):
            G1 = copy.deepcopy(graphs[i])
            G2 = copy.deepcopy(graphs[j])
            res = WL(G1, G2, 7).train()
            # print(f"{res}, {G1.label}, {G2.label}")
            if G1.label == G2.label:
                if G1.label == 0:
                    a00 += res
                    count_00 += 1
                else:
                    a11 += res
                    count_11 += 1
            else:
                a01 += res
                count_01 += 1

    print(
        f"Average similarity for graphs with labels 0 = {(a00 / count_00) / x1 ** 2}")
    print(
        f"Average similarity for graphs with labels 1 = {(a11 / count_11) / y1 ** 2}")
    print(
        f"Average similarity for graphs with diff labels = {(a01 / count_01) / ((x1 * x2 + y1 * y2)/ (x2 + y2)) ** 2}")

    print(
        f"Accuracy on testing {accuracy(graphs_test, graphs, avg_nodes_0=x1, avg_nodes_1=y1)}")


if __name__ == "__main__":
    main(100)
