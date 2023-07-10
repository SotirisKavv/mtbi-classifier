import numpy as np
import networkx as nx

from utils.timer import timer


def orthogonal_minimum_spanning_tree(graph, k):
    """
    graph: adjacency matrix
    k: mst rounds
    """

    original = graph.copy()
    graph = 1 - graph

    while k > 0:
        cur_mst = calculate_mst(graph)
        graph[cur_mst > 0] = np.inf
        k -= 1

    omst = original * np.where(graph > 1, 1, 0)

    return omst


def calculate_mst(adjacency_matrix):
    graph = nx.Graph(adjacency_matrix)
    mst = nx.minimum_spanning_tree(graph)

    return nx.to_numpy_array(mst)
