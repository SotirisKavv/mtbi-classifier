import networkx as nx
import numpy as np
import pandas as pd
from community import community_louvain

bar_len = 20


def extract_features(graphs, labels, mode="BC"):
    cols = []

    if mode == "NS":
        cols = ["NS{:0>3}".format(i) for i in range(len(graphs[0]))]
    elif mode == "BC":
        cols = ["BC{:0>3}".format(i) for i in range(len(graphs[0]))]
    elif mode == "FE":
        cols = [
            "node_strengths_mean",
            "charachteristic_path_length",
            "global_efficiency",
            "centrality",
            "modularity",
            "participation_coefficient",
            "assortativity",
        ]

    features = pd.DataFrame(columns=cols)

    for i, graph in enumerate(graphs):
        print(
            "Processing Graphs   [{}{}] {}/{}".format(
                "█" * int(bar_len * (i / len(graphs))),
                "-" * (bar_len - int(bar_len * (i / len(graphs)))),
                i + 1,
                len(graphs),
            ),
            end="\r",
        )

        if mode == "NS":
            feature_vector = node_strengths_features(graph)
        elif mode == "BC":
            feature_vector = participation_coefficients_features(graph)
        else:
            feature_vector = graph_features(graph)

        features.loc[i] = feature_vector

        print(end="\x1b[2K")
    print(
        "Extraction Complete [{}] {}/{}".format(
            "█" * bar_len,
            len(graphs),
            len(graphs),
        )
    )
    features["label"] = labels

    return features


def node_strengths_features(adj_matrix):
    G = nx.from_numpy_array(adj_matrix)
    G.remove_edges_from(nx.selfloop_edges(G))

    feature_vector = np.array(
        [G.degree(weight="weight")[node] for node in G.nodes]
    )  # Node Strengths

    return feature_vector


def participation_coefficients_features(adj_matrix):
    G = nx.from_numpy_array(adj_matrix)
    G.remove_edges_from(nx.selfloop_edges(G))

    # partition = community_louvain.best_partition(G)

    # Calculate Participation Coefficientσ
    feature_vector = list(nx.betweenness_centrality(G, weight="weight").values())

    return feature_vector


def graph_features(adj_matrix):
    G = nx.from_numpy_array(adj_matrix)
    G.remove_edges_from(nx.selfloop_edges(G))

    # Community detection using the Louvain method
    partition = community_louvain.best_partition(G)

    # Calculate Participation Coefficient
    participation_coefficient = []
    for node in G.nodes:
        node_community = partition[node]
        node_degree = G.degree(node)
        intra_community_degree = sum(
            1 for neighbor in G.neighbors(node) if partition[neighbor] == node_community
        )
        participation_coeff = 1 - (intra_community_degree / node_degree) ** 2
        participation_coefficient.append(participation_coeff)

    feature_vector = np.array(
        [
            np.array(
                [G.degree(weight="weight")[node] for node in G.nodes]
            ).mean(),  # Node Strengths
            nx.average_shortest_path_length(
                G, weight="weight"
            ),  # Characteristic Path Length
            nx.global_efficiency(G),  # Global Efficiency
            np.mean(
                list(nx.betweenness_centrality(G, weight="weight").values())
            ),  # Betweenness Centrality
            community_louvain.modularity(partition, G),  # Modularity
            np.mean(participation_coefficient),  # Participation Coefficient
            nx.degree_assortativity_coefficient(G, weight="weight"),  # Assortativity
        ]
    )

    return feature_vector
