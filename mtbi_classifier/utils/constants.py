#  import functional connectivity estimators
from festimators.mi import mutual_information as mi
from festimators.iplv import imag_phase_lock_value as iplv
from festimators.aec import amplitude_envelope_correlation as aec

METHODS = [("IPLV", iplv), ("MI", mi), ("AEC", aec)]

SAMPLE_RATE = 1017.25  # Hz
OMST_LEVEL = 15

# low, high, order
DELTA_BAND = (0.5, 4, 3)
THETA_BAND = (4, 8, 4)
ALPHA_BAND = (8, 13, 4)
BETA_BAND = (13, 30, 6)
GAMMA_BAND = (30, 70, 8)

BANDS = [DELTA_BAND, THETA_BAND, ALPHA_BAND, BETA_BAND, GAMMA_BAND]
BAND_NAMES = ["DELTA_BAND", "THETA_BAND", "ALPHA_BAND", "BETA_BAND", "GAMMA_BAND"]
