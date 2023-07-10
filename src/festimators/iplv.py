import numpy as np
from scipy.signal import hilbert


# implement imaginary phaselocking value (iPLV) for static functional brain networks
def imag_phase_lock_value(multi1, multi2):
    """
    A fast implementation of PLV for static functional brain networks
    INPUT : multi1, multi2 = filtered multichannel recordings with dimensions
                    rois x samples (time points)
    OUTPUT : iplv = rois x rois
    """
    samples, rois = multi1.shape

    Q1 = np.exp(1j * np.angle(hilbert(multi1)))
    Q2 = np.exp(1j * np.angle(hilbert(multi2)))
    iplv = np.zeros((rois, rois))

    for i in range(rois):
        Q1_i = Q1[:, i]
        iphase_lock = np.abs(
            np.imag(np.sum(Q2 / Q1_i[:, np.newaxis], axis=0)) / samples
        )
        iplv[i, :] = iphase_lock
        iplv[:, i] = iphase_lock

    return (iplv - np.min(iplv)) / (np.max(iplv) - np.min(iplv))
