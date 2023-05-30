# classification using graph embedding through Laplacian Decomposition
import networkx as nx
import numpy as np
from scipy.linalg import eigh
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


def classifyLD(data, labels):
    # Convert your graphs to the format required by NetworkX if needed

    # Determine the number of nodes
    n_nodes = data[0].number_of_nodes()

    # Prepare for Laplacian decomposition
    graph_embeddings = []

    for graph in data:
        # Calculate the Laplacian matrix
        L = nx.normalized_laplacian_matrix(graph).todense()

        # Perform eigen-decomposition and take the eigenvectors corresponding to the 2nd to (n_nodes+1)th smallest eigenvalues
        # We avoid the smallest eigenvalue as it corresponds to trivial solution
        _, eigenvectors = eigh(L)
        graph_embedding = eigenvectors[:, 1 : n_nodes + 1].flatten()
        graph_embeddings.append(graph_embedding)

    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        graph_embeddings, labels, test_size=0.2, random_state=42
    )

    # Train an SVM using the graph embeddings
    clf = SVC()
    clf.fit(X_train, y_train)

    # Make predictions
    predictions = clf.predict(X_test)

    return y_test, predictions
