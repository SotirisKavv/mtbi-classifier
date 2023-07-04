import matplotlib.pyplot as plt
import numpy as np

## used for plotting
# from scipy.fft import fft, fftfreq, fftshift
from scipy.signal import butter, lfilter


def bandpass(x, fs, band, order):
    y = np.zeros(x.shape)

    for i in range(x.shape[1]):
        y[:, i] = butter_bp_filter(x[:, i], band[0], band[1], fs, order)

    return y


def butter_bp(l, h, fs, order):
    return butter(order, [l, h], btype="bandpass", fs=fs)


def butter_bp_filter(data, l, h, fs, order):
    b, a = butter_bp(l, h, fs, order=order)
    y = lfilter(b, a, data)
    return y
