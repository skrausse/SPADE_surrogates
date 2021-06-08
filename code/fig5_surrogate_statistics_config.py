"""
Configurations to create the figure containing the overview over statistical
features of the different surrogate methods.
"""
import os

import numpy as np
import quantities as pq

from generate_artificial_data import get_shape_factor_from_cv2

DATA_PATH = '../data/surrogate_statistics'
PLOT_PATH = '../plots'

if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)
if not os.path.exists(PLOT_PATH):
    os.makedirs(PLOT_PATH)

FIG_NAME = 'fig5_surrogate_statistics.eps'

#  data generation

# Global surrogate methods and spike train types.
SURR_METHODS = ('UD', 'UDD', 'JISI-D', 'ISI-D', 'SHIFT-ST', 'BIN-SHUFF')
DATA_TYPES = ('Poisson', 'PPD', 'Gamma')

# Data type for firing rate step and for eff. moved
STEP_DATA_TYPE = 'Gamma'

# firing rate for ISI-distribtion, autocorrelation, cross-correlation,
# cv-change
FIRING_RATE = 60.*pq.Hz
# rates for clipped firing rates and eff. moved
RATES = np.arange(10., 100.1, 10.) * pq.Hz
# Firing rates for step
FIRING_RATES_STEP = (10 * pq.Hz, 80 * pq.Hz)
# DURATION_RATES_STEP
DURATION_RATES_STEP = 150. * pq.ms
# number of spikes for ISI-distribtion, autocorrelation, cross-correlation
HIGH_NUMBER_SPIKES = 500000
# number of spikes for clipped rates, effective moved, CV change
LOW_NUMBER_SPIKES = 10000
# number spiketrains for firing rate step
NUMBER_SPIKETRAINS = 10000

# CVS for CV change
CVS = np.arange(0.4, 1.21, 0.05)

# dither-parameter
DITHER = 25 * pq.ms
# SPADE bin size
SPADE_BIN_SIZE = 5. * pq.ms

# bin size for ISI-distribtion, autocorrelation, cross-correlation
BIN_SIZE = 1. * pq.ms
# number of bins auto-/cross-correlation
NUM_BINS = 60
# ISI limit in terms of mean ISI
ISI_LIM = 3.

# Parameters Trial shifting
TRIAL_LENGTH = 500. * pq.ms
TRIAL_SEPARATION = 0. * pq.ms

# Values for refractory period and CV2
DEAD_TIME = 1.6 * pq.ms
CV2 = 0.75

SHAPE_FACTOR = get_shape_factor_from_cv2(CV2)


#  plotting

FIGSIZE = (7.5, 8.75)
XLABELPAD = -0.5
YLABELPAD = -0.5
ORIGINAL_LINEWIDTH = 2.2
SURROGATES_LINEWIDTH = 0.75
FONTSIZE = 10

LABELS = {'original': 'original',
          'UD': 'UD',
          'UDD': 'UDD',
          'ISI-D': 'ISI-D',
          'JISI-D': 'JISI-D',
          'SHIFT-ST': 'TR-SHIFT',
          'BIN-SHUFF': 'WIN-SHUFF'}

COLORS = {'original': 'C0',
          'UD': 'C1',
          'UDD': 'C2',
          'ISI-D': 'C4',
          'JISI-D': 'C6',
          'SHIFT-ST': 'C3',
          'BIN-SHUFF': 'C5'}

LINE_STYLES = {'original': 'solid',
               'UD': 'solid',
               'UDD': 'solid',
               'ISI-D': 'dashed',
               'JISI-D': 'solid',
               'SHIFT-ST': 'solid',
               'BIN-SHUFF': 'solid'}

# Panel letters
LETTERS = ('A', 'B', 'C', 'D', 'E', 'F', 'G')

# limits AC/CC relative to rate
AC_BOTTOM = 0.8
AC_TOP = 1.1
CC_BOTTOM = 0.8
CC_TOP = 1.6

# AC/ CC xlim relative to dither
AC_CC_XLIM = 2.1
