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
    samples, rois = multi.shape

    Q = np.exp(1j * np.angle(hilbert(multi)))
    iplv = np.zeros((rois, rois))

    for i in range(rois):
        for j in range(rois):
            iphase_lock = np.abs(np.imag(np.sum(Q[:, j] / Q[:, i])) / samples)
            iplv[i, j] = iphase_lock
            iplv[j, i] = iphase_lock

    return (iplv - np.min(iplv)) / (np.max(iplv) - np.min(iplv))
