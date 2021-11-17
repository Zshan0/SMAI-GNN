import sys

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


def main():
    graphs, classes_count = parse_dataset("PROTEINS")
    graphs = [
        graphs[random.randint(0, len(graphs) - 1)]
        for _ in range(100)
    ]

    """
    Average similarity for similar graphs = x
    Average similarity for dissimilar graphs = y    
    """
    x = 0
    y = 0
    count_x = 0
    count_y = 0
    for i in range(len(graphs)):
        for j in range(i + 1, len(graphs)):
            G1 = copy.deepcopy(graphs[i])
            G2 = copy.deepcopy(graphs[j])
            res = WL(G1, G2, 10).train()
            # print(f"{res}, {G1.label}, {G2.label}")
            if G1.label == G2.label:
                x += res
                count_x += 1
            else:
                y += res
                count_y += 1

    print(f"Average similarity for similar graphs = {x / count_x}")
    print(f"Average similarity for dissimilar graphs = {y / count_y}")


main()
