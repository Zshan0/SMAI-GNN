import sys

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

graphs, classes_count = parse_dataset("PROTEINS")


def wl_func(G1,G2):
    wl = WL(G1, G2, 10)
    result = wl.train()
    return result

n = 5

for i in range(n):
    for j in range(i+1,n):
        print(f"Running WL Test on Graph {i} and {j}")
        print(f"Final Result of Graph {i} and {j}: {wl_func(graphs[i],graphs[j])} with labels {graphs[i].label} and {graphs[j].label}")