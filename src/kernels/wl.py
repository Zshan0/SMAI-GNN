import sys
from typing import List, Optional

try:
    from graph import Graph
except:
    sys.path.append("..")
    from graph import Graph


class WL:
    """
    The Weisfeiler-Lehman Isomorphism Test
    """

    def __init__(self, G1: Graph, G2: Graph, num_iter):
        self.G1 = G1
        self.G2 = G2
        self.nodes1 = len(self.G1.g)
        self.nodes2 = len(self.G2.g)
        self.niter = num_iter
        self.max_label = 0
        self.label1 = self.G1.node_features
        self.label2 = self.G2.node_features
        self.M1 = [0] * self.nodes1
        self.M2 = [0] * self.nodes2

    def unite(self, vec):
        """
        Uniting n-dimensional feature vector of each node into a single value.
        We need a injective function to unite the feature vectors to a single value.
        Approaches:
            - concatenation
            - binary
            - index of one
        """
        pass

    def multisetlabel_determination(self):
        """
        Multiset-label determination
            1. For i = 0, set M_i(v) := l_0(v) = l(v)
            2. For i > 0, assign a multiset-label M_i(v) to
            each node v in G and G′ which consists of the multiset {l_{i − 1}(u) | u \in N(v)}
        """
        pass

    def sorting(self):
        """
        Sort elements in M_i(v) in ascending order and concatenate them into a string s_i(v).
        Add l_{i − 1}(v) as a prefix to s_i(v) and call the resulting string s_i(v).
        """
        pass

    def label_compression(self, string_repr1, string_repr2):
        """
        Compression:
            1. Sort all of the strings s_i(v) for all v from G and G′ in ascending order.
            2. Map each string s_i(v) to a new compressed label, using a function f : Σ∗ → Σ such that
            f_(s_i(v)) = f_(s_i(w)) if and only if s_i(v) = s_i(w).
        Relabeling:
            Set l_i(v) := f_(s_i(v)) for all nodes in G and G′.
        """
        pass

    def kernel(self):
        """
        At ith iteration, we have node_features for nodes of Graph 1 and Graph 2.
        We use them to calculate similarity between the two graphs.
            - similarity += f(occurrences_in_graph_1[label], occurrences_in_graph_2[label])
                for each label
            - similarity /= f(self.nodes1, self.nodes2)
        Here, f is a function that takes two arguments and returns a single value.
        It can be:
            - multiplication
            - min
        """
        pass

    def train(self):
        similarity = (self.niter + 1) * self.kernel()
        for i in range(self.niter):
            self.multisetlabel_determination()
            s1, s2 = self.sorting()
            self.label_compression(s1, s2)
            current_similarity = self.kernel()
            similarity += current_similarity * (self.niter - i)

        return similarity
