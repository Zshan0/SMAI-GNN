import copy
from pprint import pprint as pp
import networkx as nx
import sys
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.defchararray import count
try:
    from graph import Graph
    from data_handling import *
except:
    sys.path.append("..")
    from graph import Graph
    from data_handling import *


def analyse(graphs, classes_count):
    data1 = {
        "min nodes in one graph": np.inf,
        "max nodes in one graph": 0,
        "total nodes for all graphs": 0,
        "number of graphs with this label": 0,
        "average nodes": 0
    }
    data2 = {
        "min nodes in one graph": np.inf,
        "max nodes in one graph": 0,
        "total nodes for all graphs": 0,
        "number of graphs with this label": 0,
        "average nodes": 0
    }
    stats = [data1, data2]
    for G in graphs:
        stats[G.label]["min nodes in one graph"] = min(
            stats[G.label]["min nodes in one graph"], len(G.g))
        stats[G.label]["max nodes in one graph"] = max(
            stats[G.label]["max nodes in one graph"], len(G.g))
        stats[G.label]["total nodes for all graphs"] += len(G.g)
        stats[G.label]["number of graphs with this label"] += 1
        stats[G.label]["average nodes"] = stats[G.label]["total nodes for all graphs"] / \
            stats[G.label]["number of graphs with this label"]

    pp(stats)

    count_0 = 0
    count_1 = 0
    for G in graphs:
        if G.label == 0:
            count_0 += 1
        else:
            count_1 += 1
    print("Number of graphs with label 0:", count_0)
    print("Number of graphs with label 1:", count_1)

    return stats[0]["average nodes"], stats[0]["total nodes for all graphs"], \
        stats[1]["average nodes"], stats[1]["total nodes for all graphs"]
