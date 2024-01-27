import networkx as nx
import numpy as np
import pandas as pd
from community import community_louvain

bar_len = 20


def extract_features_from_graphs(graphs, labels, mode="BC"):
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
            "Processing Graphs   |{}{}| {}/{}".format(
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
            feature_vector = betweenness_features(graph)
        else:
            feature_vector = graph_features(graph)

        features.loc[i] = feature_vector

        print(end="\x1b[2K")
    print(
        "Extraction Complete |{}| {}/{}".format(
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


def betweenness_features(adj_matrix):
    G = nx.from_numpy_array(adj_matrix)
    G.remove_edges_from(nx.selfloop_edges(G))

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


def exctract_features_from_signals(signal):
    def curve_length(x):
        return np.sum(np.abs(np.diff(x)))

    def zero_crossings(x):
        return np.sum(np.diff(np.sign(x)) != 0)

    def intensity_weighted_mean_freq_bw(x):
        freq_values = np.fft.rfftfreq(len(x))
        psd_values = np.abs(np.fft.rfft(x)) ** 2
        # Calculate mean frequency
        mean_frequency = np.sum(freq_values * psd_values) / np.sum(psd_values)
        # Calculate bandwidth
        bandwidth = np.sqrt(
            np.sum((freq_values - mean_frequency) ** 2 * psd_values)
            / np.sum(psd_values)
        )
        # Calculate spectral entropy
        normalized_psd = psd_values / np.sum(psd_values)
        entropy = -np.sum(
            normalized_psd * np.log2(normalized_psd + np.finfo(float).eps)
        )

        return mean_frequency, bandwidth, entropy

    def absolute_area(x):
        return np.sum(np.abs(x))

    m_freq, bandwidth, entropy = np.apply_along_axis(
        intensity_weighted_mean_freq_bw, axis=0, arr=signal
    )

    feature_vector = np.array(
        [
            np.mean(signal, axis=0),  # Median
            np.sqrt(np.mean(np.square(signal), axis=0)),  # RMS
            np.apply_along_axis(curve_length, axis=0, arr=signal),  # Curve Length
            np.apply_along_axis(zero_crossings, axis=0, arr=signal),  # Zero Crossings
            m_freq,  # Intensity Weighted Mean Frequency
            bandwidth,  # Intensity Weighted Bandwidth
            entropy,  # Spectral Entropy
            np.apply_along_axis(absolute_area, axis=0, arr=signal),  # Absolute Area
        ]
    )

    return feature_vector


import numpy as np


def extract_psd_features(meg_signals, fs):
    """
    Extracts spectral power from MEG signals in the range of 0.5-70Hz using FFT.

    :param meg_signals: Array of MEG signals (each row is a signal)
    :param fs: Sampling frequency
    :return: Array of spectral power vectors, shape (n_signals, 70)
    """
    meg_signals = np.array(meg_signals).T
    n_signals = meg_signals.shape[0]
    n_points = meg_signals.shape[1]
    spectral_powers = np.zeros((n_signals, 70))

    freq_bins = np.fft.rfftfreq(n_points, 1 / fs)

    for i in range(n_signals):
        # Compute FFT
        fft_values = np.fft.rfft(meg_signals[i])

        # Compute magnitude spectrum
        magnitude = np.abs(fft_values)

        for j in range(70):
            # Calculate the lower and upper frequency for each unit
            lower_freq = 0.5 + j
            upper_freq = lower_freq + 1

            # Find indices of frequencies within the desired range
            freq_indices = np.where(
                (freq_bins >= lower_freq) & (freq_bins < upper_freq)
            )

            # Sum the spectral power (square of the magnitude) within the unit range
            spectral_powers[i, j] = np.sum(magnitude[freq_indices] ** 2)

    return spectral_powers
