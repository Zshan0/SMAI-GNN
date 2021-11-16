import sys
from typing import List, Optional

try:
    from graph import Graph
except:
    sys.path.append("..")
    from graph import Graph

class WL:
    """
    The Weisfeiler-Lehman Similarity Test
    """
    def __init__(self, G: Graph, num_iter):
        self.G = G
        self.nodes = len(self.G.g)
        self.iter = num_iter
        self.max_label = 0
        self.label = [1] * self.nodes
        self.M = [0] * self.nodes
    
    def multisetlabelling(self):
        """
        Assign a multiset-label M_i(v) to each node v in G which consists of 
        the multiset {l_{i − 1}(u) | u \in N(v)}
        """
        for current_node in range(self.nodes):
            multiset = []
            self.max_label = max(self.max_label, self.label[current_node])
            for neighbor in self.G.neighbors[current_node]:
                multiset.append(self.label[neighbor])
            self.M[current_node] = multiset
    
    def sorting(self):
        """
        Sort elements in M_i(v) in ascending order and concatenate them into a string s_i(v).
        Add l_{i − 1}(v) as a prefix to s_i(v).
        """
        string_repr = [""] * self.nodes
        for current_node in range(self.nodes):
            self.M[current_node].sort()
            string_repr[current_node] = str(self.label[current_node])
            for label in self.M[current_node]:
                string_repr[current_node] += str(label)
    
    def compression(self, string_repr):
        """
        Map each string s_i(v) to a compressed label using a hash function f : Σ∗ → Σ such that
        f(s_i(v)) = f(s_i(w)) if and only if s_i(v) = s_i(w)
        """
        pass

    def relabel(self):
        pass