import numpy as np
from scipy.fftpack import hilbert


def amplitude_envelope_correlation(timeseries1, timeseries2, chunk_size=1000):
    """
    Calculate the amplitude envelope correlation between X time series
    INPUT: timeseries - a NumPy array of shape (X, t) where X represents the number of time series and t is the number of time points
    OUTPUT: aec - a NumPy array of shape (X, X) containing the amplitude envelope correlation between each pair of time series
    """

    if timeseries1.shape != timeseries2.shape:
        raise ValueError(
            "Shape mismatch between timeseries1 and timeseries2: "
            f"{timeseries1.shape} vs {timeseries2.shape}"
        )

    _, X1 = timeseries1.shape
    _, X2 = timeseries2.shape

    im1 = np.apply_along_axis(hilbert, 0, timeseries1)
    im2 = np.apply_along_axis(hilbert, 0, timeseries2)

    ampl1 = np.abs(im1)
    ampl2 = np.abs(im2)

    corr = np.abs(np.corrcoef(ampl1.T, ampl2.T)[:X1, X1:])

    # Set the diagonal elements to zero before normalization
    np.fill_diagonal(corr, 0)

    return (corr - np.min(corr)) / (np.max(corr) - np.min(corr))
