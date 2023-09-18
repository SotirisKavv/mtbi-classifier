import numpy as np


def multiplex_supra_adjacency_matrix(adjacency_matrices, interlayer_edges):
    # Number of layers
    num_layers = len(adjacency_matrices)

    # Number of nodes (ROIs) in each layer
    num_nodes = adjacency_matrices[0].shape[0]

    # Initialize the supra-adjacency matrix
    supra_adjacency_matrix = np.zeros((num_nodes * num_layers, num_nodes * num_layers))

    # Fill the supra-adjacency matrix
    for i in range(num_layers):
        for j in range(num_layers):
            if i == j:
                # Add intra-layer edges
                supra_adjacency_matrix[
                    i * num_nodes : (i + 1) * num_nodes,
                    i * num_nodes : (i + 1) * num_nodes,
                ] = adjacency_matrices[i]
            elif i > j:
                # Add inter-layer edges
                supra_adjacency_matrix[
                    i * num_nodes : (i + 1) * num_nodes,
                    j * num_nodes : (j + 1) * num_nodes,
                ] = np.diag(np.diag(interlayer_edges[j, i]))
                supra_adjacency_matrix[
                    j * num_nodes : (j + 1) * num_nodes,
                    i * num_nodes : (i + 1) * num_nodes,
                ] = np.diag(np.diag(interlayer_edges[j, i]))

    return supra_adjacency_matrix


def multilayer_supra_adjacency_matrix(adjacency_matrices, interlayer_edges):
    # Number of layers
    num_layers = len(adjacency_matrices)

    # Number of nodes (ROIs) in each layer
    num_nodes = adjacency_matrices[0].shape[0]

    # Initialize the supra-adjacency matrix
    supra_adjacency_matrix = np.zeros((num_nodes * num_layers, num_nodes * num_layers))

    # Fill the supra-adjacency matrix
    for i in range(num_layers):
        for j in range(num_layers):
            if i == j:
                # Add intra-layer edges
                supra_adjacency_matrix[
                    i * num_nodes : (i + 1) * num_nodes,
                    i * num_nodes : (i + 1) * num_nodes,
                ] = adjacency_matrices[i]
            elif i > j:
                # Add inter-layer edges
                supra_adjacency_matrix[
                    i * num_nodes : (i + 1) * num_nodes,
                    j * num_nodes : (j + 1) * num_nodes,
                ] = interlayer_edges[j, i]
                supra_adjacency_matrix[
                    j * num_nodes : (j + 1) * num_nodes,
                    i * num_nodes : (i + 1) * num_nodes,
                ] = interlayer_edges[j, i]

    return supra_adjacency_matrix


def cross_frequency_coupling(pool, method, layers):
    # Calculate the interlayer adjacency matrices
    # Only the upper triangular part of the interlayer adjacency matrices is calculated as the matrices are symmetric

    results = [
        pool.apply_async(method, args=(layers[i], layers[j], True))
        for i in range(len(layers) - 1)
        for j in range(i + 1, len(layers))
    ]

    return [result.get() for result in results]
