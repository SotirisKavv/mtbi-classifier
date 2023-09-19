import networkx as nx
import numpy as np
import pandas as pd
from community import community_louvain

bar_len = 20


def extract_features(graphs, labels, multilayer=False):
    features = pd.DataFrame(
        columns=[
            "node_strength",
            "charachteristic_path_length",
            "global_efficiency",
            "centrality",
            "modularity",
            "participation_coefficient",
            "assortativity",
        ]
    )

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
        if multilayer:
            feature_vector = multilayer_features(graph)
        else:
            feature_vector = single_layer_features(graph)

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


def single_layer_features(adj_matrix):
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
            np.sum(
                [G.degree(weight="weight")[node] for node in G.nodes],
            ),  # Node Strength
            nx.average_shortest_path_length(
                G, weight="weight"
            ),  # Characteristic Path Length
            nx.global_efficiency(G),  # Global Efficiency
            np.mean(
                list(nx.betweenness_centrality(G, weight="weight").values())
            ),  # Betweenness Centrality
            community_louvain.modularity(partition, G),  # Modularity
            # np.mean(
            #     list(nx.rich_club_coefficient(G).values())
            # ),  # Rich Club Coefficient
            np.mean(participation_coefficient),  # Participation Coefficient
            nx.degree_assortativity_coefficient(G, weight="weight"),  # Assortativity
        ]
    ).flatten()

    return feature_vector


def multilayer_features(adj_matrix):
    num_layers = adj_matrix.shape[0]

    all_node_strengths = []
    all_participation_coefficients = []
    characteristic_path_length = []
    global_efficiency = []
    betweenness_centrality = []
    modularity = []
    rich_club_coefficient = []
    assortativity = []

    for layer in range(num_layers):
        G = nx.from_numpy_matrix(adj_matrix[layer])

        # Community detection using the Louvain method
        partition = community_louvain.best_partition(G)

        # Calculate Participation Coefficient
        participation_coefficient = []
        for node in G.nodes:
            node_community = partition[node]
            node_degree = G.degree(node)
            intra_community_degree = sum(
                1
                for neighbor in G.neighbors(node)
                if partition[neighbor] == node_community
            )
            participation_coeff = 1 - (intra_community_degree / node_degree) ** 2
            participation_coefficient.append(participation_coeff)

        # Node Strengths
        node_strengths = [G.degree(weight="weight")[node] for node in G.nodes]

        all_node_strengths.append(node_strengths)
        all_participation_coefficients.append(participation_coefficient)
        characteristic_path_length.append(
            nx.average_shortest_path_length(G, weight="weight")
        )
        global_efficiency.append(nx.global_efficiency(G))
        betweenness_centrality.append(
            np.mean(list(nx.betweenness_centrality(G, weight="weight").values()))
        )
        modularity.append(community_louvain.modularity(partition, G))
        rich_club_coefficient.append(
            np.mean(list(nx.rich_club_coefficient(G).values()))
        )
        assortativity.append(nx.degree_assortativity_coefficient(G, weight="weight"))

    feature_vector = [
        all_node_strengths,  # Node Strengths for each layer
        np.mean(
            characteristic_path_length
        ),  # Average Characteristic Path Length across layers
        np.mean(global_efficiency),  # Average Global Efficiency across layers
        np.mean(betweenness_centrality),  # Average Betweenness Centrality across layers
        np.mean(modularity),  # Average Modularity across layers
        np.mean(rich_club_coefficient),  # Average Rich Club Coefficient across layers
        all_participation_coefficients,  # Participation Coefficient for each layer
        np.mean(assortativity),  # Average Assortativity across layers
    ]

    return feature_vector
