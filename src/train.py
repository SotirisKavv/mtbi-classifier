import time
from scipy import io as sio
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils.filter import bandpass
from utils.constants import *
from festimators.iplv import imag_phase_lock_value as iplv
from festimators.mi import mutual_information as mi
from festimators.aec import amplitude_envelope_correlation as aec
from utils.mstk import orthogonal_minimum_spanning_tree as omst


def run():
    print("Loading data...")
    data = sio.loadmat("data/mTBI/sources_TBI_MEGM001.mat")
    meg = pd.DataFrame(data["interp"].T)
    meg.columns = [
        label[0] for label in np.ndarray.flatten(data["virtualdata"]["label"][0][0])
    ]
    print("Data Loaded successfully.")

    rois = len(meg.columns)

    aggr_iplv = np.zeros((rois, rois))
    aggr_mi = np.zeros((rois, rois))
    aggr_aec = np.zeros((rois, rois))

    for band in BANDS:
        print("==== band {} - {} ====".format(band[0], band[1]))
        filt_meg = bandpass(
            meg.to_numpy(),
            SAMPLE_RATE,
            band,
            band[2],
        )

        fmap_iplv = iplv(filt_meg.T)
        fmap_mi = mi(filt_meg)
        fmap_aec = aec(filt_meg)

        aggr_iplv += fmap_iplv
        aggr_mi += fmap_mi
        aggr_aec += fmap_aec

    # Single Layer FC
    aggr_iplv /= len(BANDS)
    aggr_mi /= len(BANDS)
    aggr_aec /= len(BANDS)

    plt.figure(figsize=(12, 4))
    plt.suptitle("Aggregated results")

    plt.subplot(131)
    sns.heatmap(aggr_iplv, cmap="viridis")
    plt.title("IPLV")
    plt.xlabel("ROIs")
    plt.ylabel("ROIs")

    plt.subplot(132)
    sns.heatmap(aggr_mi, cmap="viridis")
    plt.title("MI")
    plt.xlabel("ROIs")
    plt.ylabel("ROIs")
    plt.tight_layout()

    plt.subplot(133)
    sns.heatmap(aggr_aec, cmap="viridis")
    plt.title("AEC")
    plt.xlabel("ROIs")
    plt.ylabel("ROIs")
    plt.tight_layout()

    mst_iplv = omst(aggr_iplv, 3)
    mst_mi = omst(aggr_mi, 3)
    mst_aec = omst(aggr_aec, 3)

    plt.figure(figsize=(12, 4))
    plt.suptitle("3-MST results")

    plt.subplot(131)
    sns.heatmap(mst_iplv, cmap="viridis")
    plt.title("IPLV")
    plt.xlabel("ROIs")
    plt.ylabel("ROIs")

    plt.subplot(132)
    sns.heatmap(mst_mi, cmap="viridis")
    plt.title("MI")
    plt.xlabel("ROIs")
    plt.ylabel("ROIs")
    plt.tight_layout()

    plt.subplot(133)
    sns.heatmap(mst_aec, cmap="viridis")
    plt.title("AEC")
    plt.xlabel("ROIs")
    plt.ylabel("ROIs")
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    run()
