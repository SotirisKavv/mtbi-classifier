#   Example 01: Filtering
# from import_dir import setup_package

# setup_package()

import matplotlib.pyplot as plt

#  import Utils
from utils.constants import SAMPLE_RATE, BANDS
from utils.import_data import import_panda_csv
from filter.filter import bandpass
from utils.plot_fig import plot_fig


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
    titles.append("{}-{}Hz".format(band[0], band[1]))

#  plot original signal and filtered signals
plot_fig(
    "ps_single",
    (7, 6),
    meg["Precentral_L"].values,
    "Original Signal Spectrum",
    "Frequency [Hz]",
    "Magnitude",
)

plot_fig(
    "ps_multiple",
    (16, 8),
    [filt_sig.T[0] for filt_sig in filtered],
    title=titles,
    xlabel="Frequency [Hz]",
    ylabel="Magnitude",
    suptitle="Filtered Signals Spectra",
)

plt.show()
