#  Main file for training the model
#  import py libraries
import os
import numpy as np
import pandas as pd
import multiprocessing

# from numba import jit, cuda
from scipy import io as sio
import matplotlib.pyplot as plt
import networkx as nx

#  import Utils
from utils.timer import timer
from utils.constants import *
from utils.filter import bandpass
from classification.index import classify
from utils.plot_fig import plot_fig, plot_circular_chord, show_brain_connectivity
from utils.node_coordinates import get_coordinates
from utils.mstk import orthogonal_minimum_spanning_tree as omst
from utils.multiplex import (
    multiplex_supra_adjacency_matrix as mpsam,
    multilayer_supra_adjacency_matrix as mlsam,
    cross_frequency_coupling as cfc,
)


def test_classification():
    graphs = []
    labels = []
    for root, _, file in os.walk("data"):
        for f in file:
            print("Processing file {}".format(f))
            label = 1 if "TBI" in f else 0
            meg = import_panda_csv(os.path.join(root, f))
            g = test_pipline(meg)
            graphs.append(g)
            labels.append(label)
            print("=========================================")
            print("\r", end="")
    print("Graphs loaded successfully!")
    print("Starting classification...")
    classify(graphs, labels, "CNN")
    plt.show()


def create_graphs():
    for root, _, file in os.walk("data"):
        for f in file:
            print("Processing file {}".format(f))
            label = 1 if "TBI" in f else 0
            meg = import_panda_csv(os.path.join(root, f))
            g = test_pipline(meg)
            pd.DataFrame(g).to_csv(
                "data/single/IPLV/{}{}_graph.csv".format(
                    "TBI" if label == 1 else "HC", f.split(".")[0][-2:]
                )
            )
            print("=========================================")


# @jit(target_backend="cuda")
def test_pipline(meg):
    pool = multiprocessing.Pool()

    print("Starting filtering...", end="\r")
    filtered_signals = filter(meg, SAMPLE_RATE, BANDS)
    print(end="\x1b[2K")

    # #   create functional connectivity maps using festimator
    # print("Creating functional connectivity maps using IPLV...", end="\r")
    # fc_maps = create_fcmap(pool, iplv, filtered_signals)
    # print(end="\x1b[2K")

    # #   calculate interlayer weights maps using festimator
    # print("Calculating interlayer weights using {}...".format("IPLV"), end="\r")
    # triu_fc = create_interlayer_fcmaps(pool, iplv, filtered_signals)
    # print(end="\x1b[2K")

    # #   3rd topology: Full MultiLayer FC - Full Multilayer network
    # print("\rApplying Full Multilayer...", end="\r")
    # graph = multilayer_architecture(fc_maps, triu_fc, "IPLV")
    # print(end="\x1b[2K")

    # #   create functional connectivity maps using festimator
    # print("Creating functional connectivity maps using MI...", end="\r")
    # fc_maps = create_fcmap(pool, mi, filtered_signals)
    # print(end="\x1b[2K")

    # #   calculate interlayer weights maps using festimator
    # print("Calculating interlayer weights using {}...".format("MI"), end="\r")
    # triu_fc = create_interlayer_fcmaps(pool, iplv, filtered_signals)
    # print(end="\x1b[2K")

    # #   3rd topology: Full MultiLayer FC - Full Multilayer network
    # print("\rApplying Full Multilayer...", end="\r")
    # graph = multilayer_architecture(fc_maps, triu_fc, "MI")
    # print(end="\x1b[2K")

    # #   create functional connectivity maps using festimator
    # print("Creating functional connectivity maps using AEC...", end="\r")
    # fc_maps = create_fcmap(pool, aec, filtered_signals)
    # print(end="\x1b[2K")

    # #   calculate interlayer weights maps using festimator
    # print("Calculating interlayer weights using {}...".format("AEC"), end="\r")
    # triu_fc = create_interlayer_fcmaps(pool, iplv, filtered_signals)
    # print(end="\x1b[2K")

    # #   3rd topology: Full MultiLayer FC - Full Multilayer network
    # print("\rApplying Full Multilayer...", end="\r")
    # graph = multilayer_architecture(fc_maps, triu_fc, "AEC")
    # print(end="\x1b[2K")

    #   1st topology: Single Layer FC - Aggregate FC Maps into one for each sample
    #   create functional connectivity maps using festimator
    print("Creating functional connectivity maps using MI...", end="\r")
    fc_maps = create_fcmap(pool, mi, filtered_signals)
    print(end="\x1b[2K")

    print("\rApplying Single Layer FC...", end="\r")
    graph = single_layer_architecture(fc_maps, "MI")
    print(end="\x1b[2K")

    # #   2nd topology: Multiplex FC - Usse the supradjaceny matrix to create a multiplex network
    #   create functional connectivity maps using festimator
    # print("Creating functional connectivity maps using AEC...", end="\r")
    # fc_maps = create_fcmap(pool, aec, filtered_signals)
    # print(end="\x1b[2K")

    #    calculate interlayer weights maps using festimator
    # print("Calculating interlayer weights using {}...".format("AEC"), end="\r")
    # triu_fc = create_interlayer_fcmaps(pool, iplv, filtered_signals)
    # print(end="\x1b[2K")

    # print("\rApplying Multiplex...", end="\r")
    # graph = multiplex_architecture(fc_maps, triu_fc, "AEC")
    # print(end="\x1b[2K")

    print("Graph created successfully!")
    return graph


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
    aggr_iplv = np.mean(fc_maps, axis=0)
    # plot_fig(
    #     "heatmap",
    #     (5, 4),
    #     aggr_iplv,
    #     "{} Aggregated results".format(method_name),
    #     "ROIs",
    #     "ROIs",
    # )

    mst_iplv = omst(aggr_iplv, OMST_LEVEL)
    # plot_fig(
    #     "heatmap",
    #     (5, 4),
    #     mst_iplv,
    #     "{} 15-MST results".format(method_name),
    #     "ROIs",
    #     "ROIs",
    # )
    return mst_iplv


def multiplex_architecture(fc_maps, interlayer_w, method_name):
    msam_iplv = mpsam(fc_maps, interlayer_w)
    # plot_fig(
    #     "heatmap",
    #     (7, 6),
    #     msam_iplv,
    #     "{} Multiplex results".format(method_name),
    #     "ROIs",
    #     "ROIs",
    # )

    mst_msam_iplv = omst(msam_iplv, OMST_LEVEL * len(fc_maps))
    # plot_fig(
    #     "heatmap",
    #     (7, 6),
    #     mst_msam_iplv,
    #     "{} 15-MST results".format(method_name),
    #     "ROIs",
    #     "ROIs",
    # )
    return mst_msam_iplv


def multilayer_architecture(fc_maps, interlayer_w, method_name):
    msam_iplv = timer(mlsam, fc_maps, interlayer_w)
    # plot_fig(
    #     "heatmap",
    #     (7, 6),
    #     msam_iplv,
    #     "{} Multilayer results".format(method_name),
    #     "ROIs",
    #     "ROIs",
    # )

    mst_msam_iplv = timer(omst, msam_iplv, OMST_LEVEL * len(fc_maps))
    # plot_fig(
    #     "heatmap",
    #     (7, 6),
    #     mst_msam_iplv,
    #     "{} 75-MST results".format(method_name),
    #     "ROIs",
    #     "ROIs",
    # )

    return mst_msam_iplv


def filter(meg, sr, bands):
    filtered = []
    # titles = []
    for band in bands:
        filt_sig = bandpass(meg.to_numpy(), sr, (band[0], band[1]), band[2])
        filtered.append(filt_sig)
        # titles.append("{}-{}Hz".format(band[0], band[1]))
    # plot_fig(
    #     "ps_single",
    #     (7, 6),
    #     meg["Precentral_L"].values,
    #     "Original Signal Spectrum",
    #     "Frequency [Hz]",
    #     "Magnitude",
    # )
    # plot_fig(
    #     "ps_multiple",
    #     (16, 8),
    #     [filt_sig.T[0] for filt_sig in filtered],
    #     title=titles,
    #     xlabel="Frequency [Hz]",
    #     ylabel="Magnitude",
    #     suptitle="Filtered Signals Spectra",
    # )
    return filtered


def create_fcmap(pool, method, signals):
    # parallel fcmap creation
    fce_async = [pool.apply_async(method, args=(layer, layer)) for layer in signals]
    return [r.get() for r in fce_async]

    # fcmaps = []
    # for layer in signals:
    #     fc = method(layer, layer)
    #     fcmaps.append(fc)

    # return fcmaps


def create_interlayer_fcmaps(pool, method, signals):
    matrices = cfc(pool, method, signals)

    # titles = []
    # for i in range(len(BAND_NAMES) - 1):
    #     for j in range(i + 1, len(BAND_NAMES)):
    #         titles.append("{} vs {}".format(BAND_NAMES[i], BAND_NAMES[j]))

    # plot_fig(
    #     "multiple_hm", (9, 12), matrices, titles, "ROIs", "ROIs", "Interlayer Weights"
    # )
    n = len(signals)
    triu = np.empty((n, n), dtype=object)
    # Get indices for upper triangle excluding the diagonal
    indices = np.triu_indices(n, 1)
    for idx, val in zip(zip(*indices), matrices):
        triu[idx] = val
    return triu


def import_panda_csv(path):
    mylist = []

    for chunk in pd.read_csv(path, sep=";", lineterminator="\n", chunksize=5000):
        mylist.append(chunk)

    big_data = pd.concat(mylist, axis=0)
    del mylist

    return big_data


def test_single():
    meg = import_panda_csv("data\mTBI\sources_TBI_MEGM001.csv")
    graph = test_pipline(meg)
    plot_circular_chord(graph, meg.columns)
    # coords = get_coordinates(meg.columns)
    # show_brain_connectivity(graph, coords)
    plt.show()


if __name__ == "__main__":
    # run()
    # test_filt_parallel()
    # test_classification()
    test_single()
