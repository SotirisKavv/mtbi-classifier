#   Example 02: Functional Connectivity Maps
import os
import multiprocessing

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

#  import Utils
from utils.constants import SAMPLE_RATE, BANDS, OMST_LEVEL, iplv
from utils.import_data import import_all_data
from featextr.feature_extraction import extract_features
from filter.filter import bandpass
from orthogonal_mst.omst import orthogonal_minimum_spanning_tree as omst
from classification.svm import SVM


sns.set_theme(style="whitegrid")

if __name__ == "__main__":
    pool = multiprocessing.Pool()

    #  initialize lists
    # filtered = []
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
                )
            )
            #  filter data

            filt_async = [
                pool.apply_async(
                    bandpass,
                    args=(signal.to_numpy(), SAMPLE_RATE, (band[0], band[1]), band[2]),
                )
                for band in BANDS
            ]
            filtered = [r.get() for r in filt_async]

            fce_async = [
                pool.apply_async(iplv, args=(layer, layer)) for layer in filtered
            ]
            fc_maps = [r.get() for r in fce_async]

            aggr_iplv = np.mean(fc_maps, axis=0)

            mst_iplv = omst(aggr_iplv, OMST_LEVEL)
            graphs.append(mst_iplv)
            pd.DataFrame(mst_iplv).to_csv(
                "graphs/{}{:0>2}.csv".format("HC" if labels[i] == 0 else "TBI", i),
                index=False,
                sep=";",
            )

            print(end="\x1b[2K")
        print("Processing complete")

        del filtered, signals
    else:
        graphs, labels = import_all_data("graphs")
        graphs = [graph.to_numpy() for graph in graphs]

    dataset = extract_features(graphs, labels)

    X = dataset.drop("label", axis=1)
    y = dataset["label"]

    svm_classifier = SVM()

    svm_classifier.cross_validate(X, y, 5)
    plt.show()
