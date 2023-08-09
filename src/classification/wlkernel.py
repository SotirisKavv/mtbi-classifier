# classification using Weisfeiler-Lehman Subtree Kernel
from grakel import Graph, GraphKernel
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


def classifyWL(data, labels):
    # Assuming graphs is a list of your graphs, and labels is a list of your labels
    # Convert your graphs to the format required by GraKeL
    gk_graphs = [Graph(g) for g in data]
    for g in gk_graphs:
        g.construct_labels()

    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        gk_graphs, labels, test_size=0.2, random_state=42
    )

    # Initialize a Weisfeiler-Lehman subtree kernel
    # We chose a vertex histogram as the base kernel, as this is the most common choice
    # The height of the subtree is set to 1, which is the default and generally a good starting point
    # wl_kernel = GraphKernel(
    #     kernel=[
    #         {"name": "weisfeiler_lehman", "params": {"height": 1}},
    #         {"name": "vertex_histogram", "params": {}},
    #     ]
    # )
    wl_kernel = WeisfeilerLehman(
        n_iter=5, normalize=True, base_graph_kernel=VertexHistogram
    )

    # Calculate the kernel matrix
    print("WL kernel initialization")
    K_train = wl_kernel.fit_transform(X_train)
    K_test = wl_kernel.transform(X_test)

    # Train an SVM using the kernel matrix
    print("Training SVM")
    clf = SVC(kernel="precomputed")
    clf.fit(K_train, y_train)

    # Make predictions
    print("Predicting")
    predictions = clf.predict(K_test)

    return y_test, predictions
