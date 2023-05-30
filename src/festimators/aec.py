import numpy as np
from scipy.fftpack import hilbert
from sklearn.preprocessing import minmax_scale


def amplitude_envelope_correlation(timeseries):
    """
    Calculate the amplitude envelope correlation between X time series
    INPUT: timeseries - a NumPy array of shape (X, t) where X represents the number of time series and t is the number of time points
    OUTPUT: aec - a NumPy array of shape (X, X) containing the amplitude envelope correlation between each pair of time series
    """
    t, X = timeseries.shape

    im = np.array([hilbert(timeseries[:, i]) for i in range(X)])
    ampl = np.sqrt(np.transpose(timeseries) ** 2 + im**2)

    corr = np.abs(np.corrcoef(ampl))

    corr = corr - np.diag(np.diag(corr))

    return minmax_scale(corr)
