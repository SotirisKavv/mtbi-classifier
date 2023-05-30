import numpy as np
from scipy.signal import hilbert
from sklearn.preprocessing import minmax_scale


# implement imaginary phaselocking value (iPLV) for static functional brain networks
def imag_phase_lock_value(multi):
    """
    A fast implementation of PLV for static functional brain networks
    INPUT : multi = filtered multichannel recordings with dimensions
                    rois x samples (time points)
    OUTPUT : iplv = rois x rois
    """
    rois, samples = multi.shape

    Q = np.exp(1j * np.angle(hilbert(multi)))
    iplv = np.zeros((rois, rois))

    for i in range(rois):
        for j in range(rois):
            iplv[i, j] = np.abs(np.imag(np.sum(Q[i] / Q[j])) / samples)

    return minmax_scale(iplv)
