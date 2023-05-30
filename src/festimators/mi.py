import numpy as np
from sklearn.preprocessing import minmax_scale


def mutual_information(timeseries):
    """
    Calculate the mutual information between X time series
    INPUT: timeseries - a NumPy array of shape (X, t) where X represents the number of time series and t is the number of time points
    OUTPUT: mi - a NumPy array of shape (X, X) containing the mutual information between each pair of time series
    """
    t, X = timeseries.shape
    mi = np.zeros((X, X))

    for i in range(X):
        for j in range(i + 1, X):
            mutual_info = compute_mutual_information(timeseries[i], timeseries[j])
            mi[i, j] = mutual_info
            mi[j, i] = mutual_info

    return minmax_scale(mi)


def compute_mutual_information(x, y, bins=10):
    """
    Compute the mutual information between two variables using histogram-based estimation
    INPUT: x, y - one-dimensional NumPy arrays representing the variables
           bins - the number of bins for histogram estimation (default: 10)
    OUTPUT: mutual_info - the mutual information value
    """
    # Normalize the variables
    x_norm = (x - np.min(x)) / (np.max(x) - np.min(x))
    y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))

    # Calculate histograms
    hist_xy, _, _ = np.histogram2d(x_norm, y_norm, bins=bins)
    hist_x, _ = np.histogram(x_norm, bins=bins)
    hist_y, _ = np.histogram(y_norm, bins=bins)

    # Calculate probabilities
    p_xy = hist_xy / np.sum(hist_xy)
    p_x = hist_x / np.sum(hist_x)
    p_y = hist_y / np.sum(hist_y)

    # Calculate mutual information
    mutual_info = 0
    for i in range(bins):
        for j in range(bins):
            if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                mutual_info += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))

    return mutual_info
