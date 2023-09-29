import multiprocessing
import time

import seaborn as sns
import numpy as np
import pandas as pd

#  import Utils
from utils.constants import SAMPLE_RATE, BANDS, OMST_LEVEL, METHODS
from utils.import_data import import_all_data
from utils.multiplex import (
    cross_frequency_coupling as cfc,
    multiplex_supra_adjacency_matrix as msa,
    multilayer_supra_adjacency_matrix as fmsa,
)
from filter.filter import bandpass
from orthogonal_mst.omst import orthogonal_minimum_spanning_tree as omst


sns.set_theme(style="whitegrid")

if __name__ == "__main__":
    pool = multiprocessing.Pool()

    # import data from files
    signals, labels = import_all_data("data")

    for name, method in METHODS:
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

            time_1 = time.process_time()

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

            # Aggregated graph
            time_st = time.process_time()
            aggr_iplv = np.mean(fc_maps, axis=0)
            mst_iplv = omst(aggr_iplv, OMST_LEVEL)
            time_2 = time.process_time()

            pd.DataFrame(mst_iplv).to_csv(
                "graphs/aggregated/{}/{}{:0>2}.csv".format(
                    name, "HC" if labels[i] == 0 else "TBI", i
                ),
                index=False,
                sep=";",
            )

            # create supra-adjacency matrix
            time_nd = time.process_time()
            msam = msa(fc_maps, interlayer_w)
            mst_msam = omst(msam, OMST_LEVEL * len(fc_maps))
            time_3 = time.process_time()

            pd.DataFrame(mst_msam).to_csv(
                "graphs/multiplex/{}/{}{:0>2}.csv".format(
                    name, "HC" if labels[i] == 0 else "TBI", i
                ),
                index=False,
                sep=";",
            )

            # create supra-adjacency matrix
            msam = fmsa(fc_maps, interlayer_w)
            mst_msam = omst(msam, OMST_LEVEL * len(fc_maps))
            time_4 = time.process_time()

            pd.DataFrame(mst_msam).to_csv(
                "graphs/full_multilayer/{}/{}{:0>2}.csv".format(
                    name, "HC" if labels[i] == 0 else "TBI", i
                ),
                index=False,
                sep=";",
            )

            print(end="\x1b[2K")
        print(
            "Extraction Complete [{}] {}/{}".format(
                "█" * len(signals),
                len(signals),
                len(signals),
            )
        )
        print(
            "Time elapsed:\nAggregated {}s\tMultiplex {}\tFull Multilayer {}".format(
                time_2 - time_1,
                time_3 - time_2 + time_st - time_1,
                time_4 - time_3 + time_nd - time_2 + time_st - time_1,
            )
        )
    print("Processing complete")
