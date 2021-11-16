import sys
# from WLClassic import WL
from WLSubtree_Kernel import WL
import networkx as nx
from matplotlib import pyplot as plt

try:
    from graph import Graph
except:
    sys.path.append("..")
    from graph import Graph


def generate_adjacency_list(G):
    adj = [0] * len(G)
    for s, nbrs in G.adjacency():
        adj[s] = []
        for nbr in nbrs.keys():
            adj[s].append(nbr)
    return adj


G1 = nx.Graph()
G2 = nx.Graph()

# G1.add_edge(0, 1)
# G1.add_edge(0, 2)
# G1.add_edge(0, 3)
# G1.add_edge(1, 2)
# G1.add_edge(2, 4)
# G1.add_edge(3, 4)

# G2.add_edge(0, 1)
# G2.add_edge(0, 2)
# G2.add_edge(0, 4)
# G2.add_edge(1, 2)
# G2.add_edge(1, 3)
# G2.add_edge(3, 4)

G1.add_edge(0, 1)
G1.add_edge(0, 2)
G1.add_edge(1, 2)
G1.add_edge(1, 3)
G1.add_edge(2, 3)
G1.add_edge(3, 4)
G1.add_edge(3, 5)

G2.add_edge(0, 1)
G2.add_edge(1, 2)
G2.add_edge(1, 3)
G2.add_edge(1, 4)
G2.add_edge(2, 3)
G2.add_edge(3, 4)
G2.add_edge(4, 5)

plt.subplot(121)
nx.draw(G1, with_labels=True)
plt.subplot(122)
nx.draw_circular(G2, with_labels=True)
# plt.show()
plt.savefig("images/graphs2.png")


neighbors1 = [
    [1, 2, 3],
    [0, 2],
    [0, 1, 4],
    [0, 4],
    [2, 3]
]
neighbors2 = [
    [1, 2, 4],
    [0, 2, 3],
    [0, 1],
    [1, 4],
    [0, 3]
]

# print(generate_adjacency_list(G1))

neighbors1 = generate_adjacency_list(G1)
neighbors2 = generate_adjacency_list(G2)

# print(neighbors1)
# print(neighbors2)
# exit()

g1 = Graph(g=G1)
g1.neighbors = neighbors1

g2 = Graph(g=G2)
g2.neighbors = neighbors2

wl = WL(g1, g2, 10)
result = wl.train()
print(result)
