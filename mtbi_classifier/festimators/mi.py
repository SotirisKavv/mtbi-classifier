import numpy as np
from sklearn.metrics import mutual_info_score


def mutual_information(timeseries1, timeseries2, multiplex=False):
    """
    Calculate the mutual information between X time series
    INPUT: timeseries1, timeseries2 - NumPy arrays of shape (X, t) where X represents the number of time series and t is the number of time points
    OUTPUT: mi - a NumPy array of shape (X, X) containing the mutual information between each pair of time series
    """
    if timeseries1.shape != timeseries2.shape:
        raise ValueError(
            "Shape mismatch between timeseries1 and timeseries2: "
            f"{timeseries1.shape} vs {timeseries2.shape}"
        )

    _, X = timeseries1.shape
    mi = np.zeros((X, X))

    for i in range(X):
        for j in range(i + 1 if not multiplex else i, X):
            mutual_info = compute_mutual_information(
                timeseries1[:, i], timeseries2[:, j]
            )
            mi[i, j] = mutual_info
            mi[j, i] = mutual_info

    return (mi - np.min(mi)) / (np.max(mi) - np.min(mi))


def compute_mutual_information(x, y, bins=10):
    # Compute the mutual information between two variables using Scikit-learn
    mutual_info = mutual_info_score(
        None, None, contingency=np.histogram2d(x, y, bins=bins)[0]
    )
    return mutual_info
