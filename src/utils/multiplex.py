import numpy as np


#   unitary interlayer edges
def multiplex_supra_adjacency_matrix(adjacency_matrices):
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
                ] = np.identity(num_nodes)
                supra_adjacency_matrix[
                    j * num_nodes : (j + 1) * num_nodes,
                    i * num_nodes : (i + 1) * num_nodes,
                ] = np.identity(num_nodes)

    # Print the supra-adjacency matrix
    return supra_adjacency_matrix
