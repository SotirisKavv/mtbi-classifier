#   Example 02: Functional Connectivity Maps
# from import_dir import setup_package

# setup_package()

import matplotlib.pyplot as plt

#  import Utils
from utils.constants import SAMPLE_RATE, BANDS, iplv
from utils.import_data import import_panda_csv
from filter.filter import bandpass


import multiprocessing

if __name__ == "__main__":
    pool = multiprocessing.Pool()

    #  initialize lists
    filtered = []
    titles = []

    # import data
    meg = import_panda_csv("data\mTBI\sources_TBI_MEGM001.csv")

    #  filter data
    for band in BANDS:
        #  bandpass filtering
        filt_sig = bandpass(meg.to_numpy(), SAMPLE_RATE, (band[0], band[1]), band[2])

        filtered.append(filt_sig)

    fce_async = [pool.apply_async(iplv, args=(layer, layer)) for layer in filtered]
    fcmpas = [r.get() for r in fce_async]
