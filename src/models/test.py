import sys

from networkx import classes
from graph_analysis import *
import copy
import random
from WLSubtree_Kernel import WL
import networkx as nx
from matplotlib import pyplot as plt

try:
    from graph import Graph
    from data_handling import *
except:
    sys.path.append("..")
    from graph import Graph
    from data_handling import *


def generate_adjacency_list(G):
    adj = [0] * len(G)
    for s, nbrs in G.adjacency():
        adj[s] = []
        for nbr in nbrs.keys():
            adj[s].append(nbr)
    return adj


def main(num):
    """
    Average similarity for similar graphs and 
    average similarity for dissimilar graphs
    """
    graphs, classes_count = parse_dataset("PROTEINS")
    graphs = [graphs[random.randint(0, len(graphs) - 1)] for _ in range(num)]

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


main(100)
