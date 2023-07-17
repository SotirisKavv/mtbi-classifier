#  Main file for training the model
#  import py libraries
import numpy as np
import pandas as pd
import multiprocessing
from scipy import io as sio
import matplotlib.pyplot as plt

#  import Utils
from utils.timer import timer
from utils.constants import *
from utils.filter import bandpass
from utils.plot_fig import plot_fig
from utils.mstk import orthogonal_minimum_spanning_tree as omst
from utils.multiplex import multiplex_supra_adjacency_matrix as mpsam
from utils.multiplex import multilayer_supra_adjacency_matrix as mlsam
from utils.multiplex import cross_frequency_coupling as cfc


def run():
    meg = import_data("data/mTBI/sources_TBI_MEGM001.mat")

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

    mst_iplv = omst(aggr_iplv, 15)
    mst_mi = omst(aggr_mi, 15)
    mst_aec = omst(aggr_aec, 15)

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


def test_filt_parallel():
    pool = multiprocessing.Pool()
    meg = timer(import_data, "data/mTBI/sources_TBI_MEGM001.mat")

    print("Starting filtering...")
    filtered_signals = timer(filter, pool, meg, SAMPLE_RATE, BANDS)
    print("Filtering succeeded!")

    for method in METHODS:
        #   create functional connectivity maps using festimator
        print("Creating functional connectivity maps using {}...".format(method[0]))
        fc_maps = timer(create_fcmap, pool, method[1], filtered_signals)
        print("Functional connectivity maps created successfully!")

        #   calculate interlayer weights maps using festimator
        print("Calculating interlayer weights using {}...".format(method[0]))
        triu_fc = create_interlayer_fcmaps(pool, method[1], filtered_signals)
        print("Interlayer weights calculated successfully!")

        #   1st topology: Single Layer FC
        single_layer_architecture(fc_maps, method[0])

        #   2nd topology: Multiplex FC
        multiplex_architecture(fc_maps, triu_fc, method[0])

        #   3rd topology: Multilayer FC
        multilayer_architecture(fc_maps, triu_fc, method[0])

    plt.show()


def single_layer_architecture(fc_maps, method_name):
    print("Aggregating layers...")
    aggr_iplv = timer(np.mean, fc_maps, axis=0)
    plot_fig(
        "heatmap",
        (5, 4),
        aggr_iplv,
        "{} Aggregated results".format(method_name),
        "ROIs",
        "ROIs",
    )

    mst_iplv = timer(omst, aggr_iplv, OMST_LEVEL)
    plot_fig(
        "heatmap",
        (5, 4),
        mst_iplv,
        "{} 15-MST results".format(method_name),
        "ROIs",
        "ROIs",
    )


def multiplex_architecture(fc_maps, interlayer_w, method_name):
    print("Creating multiplex...")

    msam_iplv = timer(mpsam, fc_maps, interlayer_w)
    plot_fig(
        "heatmap",
        (7, 6),
        msam_iplv,
        "{} Multiplex results".format(method_name),
        "ROIs",
        "ROIs",
    )

    mst_msam_iplv = timer(omst, msam_iplv, OMST_LEVEL * len(fc_maps))
    plot_fig(
        "heatmap",
        (7, 6),
        mst_msam_iplv,
        "{} 15-MST results".format(method_name),
        "ROIs",
        "ROIs",
    )

    print("Multiplex created successfully!")


def multilayer_architecture(fc_maps, interlayer_w, method_name):
    print("Creating multilayer network...")

    msam_iplv = timer(mlsam, fc_maps, interlayer_w)
    plot_fig(
        "heatmap",
        (7, 6),
        msam_iplv,
        "{} Multilayer results".format(method_name),
        "ROIs",
        "ROIs",
    )

    mst_msam_iplv = timer(omst, msam_iplv, OMST_LEVEL * len(fc_maps))
    plot_fig(
        "heatmap",
        (7, 6),
        mst_msam_iplv,
        "{} 15-MST results".format(method_name),
        "ROIs",
        "ROIs",
    )

    print("Multiplex created successfully!")


def filter(pool, meg, sr, bands):
    filtered_async = [
        pool.apply_async(
            bandpass, args=(meg.to_numpy(), sr, (band[0], band[1]), band[2])
        )
        for band in bands
    ]
    return [r.get() for r in filtered_async]


def create_fcmap(pool, method, signals):
    fce_async = [pool.apply_async(method, args=(layer, layer)) for layer in signals]
    return [r.get() for r in fce_async]


def create_interlayer_fcmaps(pool, method, signals):
    matrices = timer(cfc, pool, method, signals)

    # titles = []
    # for i in range(len(BAND_NAMES) - 1):
    #     for j in range(i + 1, len(BAND_NAMES)):
    #         titles.append("{} vs {}".format(BAND_NAMES[i], BAND_NAMES[j]))

    # plot_fig(
    #     "multiple_hm", (9, 12), matrices, titles, "ROIs", "ROIs", "Interlayer Weights"
    # )
    n = len(signals)
    triu = np.empty((n, n), dtype=object)
    indices = np.triu_indices(
        n, 1
    )  # Get indices for upper triangle excluding the diagonal
    for idx, val in zip(zip(*indices), matrices):
        triu[idx] = val
    return triu


def import_data(path):
    print("Loading data...")
    data = sio.loadmat(path)
    meg = pd.DataFrame(data["interp"].T)
    meg.columns = [
        label[0] for label in np.ndarray.flatten(data["virtualdata"]["label"][0][0])
    ]
    print("Data Loaded successfully.")
    return meg


if __name__ == "__main__":
    # run()
    test_filt_parallel()
