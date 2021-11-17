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
        Uniting n-dimensional feature vector of each node into a single value 
        Approaches:
            - concatenation
        """
        if type(vec) == int:
            return vec
        vec = "".join(map(str, vec))
        return int(vec)

    def multisetlabel_determination(self):
        """
        Multiset-label determination
            1. For i = 0, set M_i(v) := l_0(v) = l(v)
            2. For i > 0, assign a multiset-label M_i(v) to
            each node v in G and G′ which consists of the multiset {l_{i − 1}(u) | u \in N(v)}
        """
        for current_node in range(self.nodes1):
            multiset = []
            self.max_label = max(
                self.max_label, self.unite(self.label1[current_node]))
            for neighbor in self.G1.neighbors[current_node]:
                label = self.unite(self.label1[neighbor])
                multiset.append(label)
            self.M1[current_node] = multiset

        for current_node in range(self.nodes2):
            multiset = []
            self.max_label = max(
                self.max_label, self.unite(self.label2[current_node]))
            for neighbor in self.G2.neighbors[current_node]:
                label = self.unite(self.label2[neighbor])
                multiset.append(label)
            self.M2[current_node] = multiset

    def sorting(self):
        """
        Sort elements in M_i(v) in ascending order and concatenate them into a string s_i(v).
        Add l_{i − 1}(v) as a prefix to s_i(v) and call the resulting string s_i(v).
        """
        string_repr1 = [""] * self.nodes1
        for current_node in range(self.nodes1):
            self.M1[current_node].sort()
            string_repr1[current_node] = str(
                self.unite(self.label1[current_node]))
            for label in self.M1[current_node]:
                string_repr1[current_node] += str(label)

        string_repr2 = [""] * self.nodes2
        for current_node in range(self.nodes2):
            self.M2[current_node].sort()
            string_repr2[current_node] = str(
                self.unite(self.label2[current_node]))
            for label in self.M2[current_node]:
                string_repr2[current_node] += str(label)

        return string_repr1, string_repr2

    def label_compression(self, string_repr1, string_repr2):
        """
        Compression:
            1. Sort all of the strings s_i(v) for all v from G and G′ in ascending order.
            2. Map each string s_i(v) to a new compressed label, using a function f : Σ∗ → Σ such that
            f_(s_i(v)) = f_(s_i(w)) if and only if s_i(v) = s_i(w).
        Relabeling:
            Set l_i(v) := f_(s_i(v)) for all nodes in G and G′.
        """
        dictionary1 = {}
        dictionary2 = {}
        for i in range(self.nodes1):
            if(string_repr1[i] not in dictionary1.keys()):
                dictionary1[string_repr1[i]] = []
            dictionary1[string_repr1[i]].append(i)

        for i in range(self.nodes2):
            if(string_repr2[i] not in dictionary2.keys()):
                dictionary2[string_repr2[i]] = []
            dictionary2[string_repr2[i]].append(i)

        list_of_strings = list(
            set(list(dictionary1.keys()) + list(dictionary2.keys()))
        )
        list_of_strings.sort()
        counter = self.max_label + 1
        for string in list_of_strings:
            if(string in dictionary1.keys()):
                for node in dictionary1[string]:
                    self.label1[node] = counter
            if(string in dictionary2.keys()):
                for node in dictionary2[string]:
                    self.label2[node] = counter
            counter += 1

    def print_labels(self):
        print("Graph 1")
        for node in range(self.nodes1):
            print(f"{node} : {self.label1[node]} : {self.M1[node]}")
        print("---------------")
        print("Graph 2")
        for node in range(self.nodes2):
            print(f"{node} : {self.label2[node]} : {self.M2[node]}")
        print("---------------")

    def check(self):
        label_set1 = []
        for label in self.label1:
            label_set1.append(label)
        label_set1.sort()

        label_set2 = []
        for label in self.label2:
            label_set2.append(label)
        label_set2.sort()

        if(len(label_set1) != len(label_set2)):
            return False
        for i in range(len(label_set1)):
            if(label_set1[i] != label_set2[i]):
                return False

        return True

    def kernel(self):
        dictionary1 = {}
        dictionary2 = {}
        for node in range(self.nodes1):
            label = self.unite(self.label1[node])
            if(label not in dictionary1.keys()):
                dictionary1[label] = 0
            dictionary1[label] += 1

        for node in range(self.nodes2):
            label = self.unite(self.label2[node])
            if(label not in dictionary2.keys()):
                dictionary2[label] = 0
            dictionary2[label] += 1

        similarity = 0
        for label in dictionary1.keys():
            if(label in dictionary1.keys() and label in dictionary2.keys()):
                similarity += (dictionary1[label] * dictionary2[label])

        similarity /= (self.nodes1 * self.nodes2)

        return similarity

    def train(self):
        similarity = self.niter * self.kernel()
        # print("Initial Similarity : ", similarity)
        for i in range(self.niter):
            self.multisetlabel_determination()
            s1, s2 = self.sorting()
            self.label_compression(s1, s2)
            current_similarity = self.kernel()
            # print(f"Iteration : {i} : {current_similarity}")
            # self.print_labels()
            similarity += current_similarity

        return similarity
