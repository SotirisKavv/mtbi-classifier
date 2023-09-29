#   Example 02: Functional Connectivity Maps
import os
import multiprocessing

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

#  import Utils
from utils.constants import SAMPLE_RATE, BANDS, OMST_LEVEL, aec, mi, iplv
from utils.import_data import import_all_data
from utils.multiplex import (
    cross_frequency_coupling as cfc,
    multiplex_supra_adjacency_matrix as msa,
)
from featextr.feature_extraction import extract_features
from filter.filter import bandpass
from orthogonal_mst.omst import orthogonal_minimum_spanning_tree as omst
from classification.knn import KNN


sns.set_theme(style="whitegrid")

if __name__ == "__main__":
    pool = multiprocessing.Pool()

    #  initialize lists
    if len(os.listdir("graphs")) == 0:
        graphs = []
        # import data from files
        signals, labels = import_all_data("data")

        for i, signal in enumerate(signals):
            print(
                "Processing signals [{}{}] {}/{}".format(
                    "█" * int(20 * (i / len(signals))),
                    "-" * (20 - int(20 * (i / len(signals)))),
                    i + 1,
                    len(signals),
                ),
                end="\r",
            )
            method = aec
            #  filter data
            filt_async = [
                pool.apply_async(
                    bandpass,
                    args=(signal.to_numpy(), SAMPLE_RATE, (band[0], band[1]), band[2]),
                )
                for band in BANDS
            ]
            filtered = [r.get() for r in filt_async]

            # create fc_maps
            fce_async = [
                pool.apply_async(method, args=(layer, layer)) for layer in filtered
            ]
            fc_maps = [r.get() for r in fce_async]

            # create interlayer matrices
            matrices = cfc(pool, method, filtered)

            n = len(filtered)
            interlayer_w = np.empty((n, n), dtype=object)

            indices = np.triu_indices(n, 1)
            for idx, val in zip(zip(*indices), matrices):
                interlayer_w[idx] = val

            # create supra-adjacency matrix
            msam = msa(fc_maps, interlayer_w)
            mst_msam = omst(msam, OMST_LEVEL * len(fc_maps))

            graphs.append(mst_msam)
            pd.DataFrame(mst_msam).to_csv(
                "graphs/{}{:0>2}.csv".format("HC" if labels[i] == 0 else "TBI", i),
                index=False,
                sep=";",
            )

            print(end="\x1b[2K")
        print("Processing complete")

        del filtered, signals, fc_maps, matrices, interlayer_w, msam, mst_msam
    else:
        graphs, labels = import_all_data("graphs")
        graphs = [graph.to_numpy() for graph in graphs]

    dataset = extract_features(graphs, labels, knn=True)

    X = dataset.drop("label", axis=1)
    y = dataset["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    knn = KNN()
    knn.find_best_k(X, y, range(1, 31))
    knn.train(X_train, y_train)
    knn.cross_validate(X, y, 5)

    plt.show()
