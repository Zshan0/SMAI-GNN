import sys

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

graphs, classes_count = parse_dataset("PROTEINS", False, DATA_PATH_WL)
G1 = graphs[1]
G2 = graphs[100]


def wl_func(G1, G2):
    wl = WL(G1, G2, 10)
    result = wl.train()
    return result


plt.subplot(121)
nx.draw(G1.g, with_labels=True)
plt.subplot(122)
nx.draw_circular(G2.g, with_labels=True)
plt.show()
plt.savefig("images/protein_graphs.png")


def main():
    n = 5

    for i in range(n):
        for j in range(i+1, n):
            print(f"Running WL Test on Graph {i} and {j}")
            print(
                f"Final Result of Graph {i} and {j}: {wl_func(graphs[i],graphs[j])} with labels {graphs[i].label} and {graphs[j].label}")
