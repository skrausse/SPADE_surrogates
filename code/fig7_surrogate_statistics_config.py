import os
import quantities as pq

from scipy.special import gamma as gamma_function
from scipy.optimize import brentq

DATA_PATH = '../data/surrogate_statistics'
PLOT_PATH = '../plots'

if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)
if not os.path.exists(PLOT_PATH):
    os.makedirs(PLOT_PATH)

#  data generation

# Global surrogate methods and spike train types.
SURR_METHODS = ('UD', 'UDR', 'ISI-D', 'JISI-D', 'SHIFT-ST', 'BIN-SHUFF')
DATA_TYPES = ('Poisson', 'PPD', 'Gamma')

# dither-parameter
DITHER = 25 * pq.ms

# Values for refractory period and CV2
DEAD_TIME = 1.6 * pq.ms
CV2 = 0.75


def get_cv2_from_shape_factor(shape_factor):
    """
    calculates cv2 from shape factor based on van Vreswijk's formula

    Parameters
    ----------
    shape_factor : float

    Returns
    -------
    cv2 : float
    """
    return gamma_function(2 * shape_factor) / \
        (shape_factor * (2 ** (shape_factor - 1)
                         * gamma_function(shape_factor))) ** 2


def get_shape_factor_from_cv2(cv2):
    """
    calculates shape factor from cv2 based on van Vreswijk's formula

    Parameters
    ----------
    cv2 : float

    Returns
    -------
    shape_factor : float
    """

    return brentq(lambda x: get_cv2_from_shape_factor(x) - cv2, 0.05, 50.)


SHAPE_FACTOR = get_shape_factor_from_cv2(CV2)


#  plotting

XLABELPAD = -0.5
YLABELPAD = -0.5
ORIGINAL_LINEWIDTH = 2.2


LABELS = {'original': 'original',
          'UD': 'UD',
          'UDR': 'UDD',
          'ISI-D': 'ISI-D',
          'JISI-D': 'JISI-D',
          'SHIFT-ST': 'TR-SHIFT',
          'BIN-SHUFF': 'BIN-SHUFF'}

COLORS = {'original': 'C0',
          'UD': 'C1',
          'UDR': 'C2',
          'ISI-D': 'C4',
          'JISI-D': 'C6',
          'SHIFT-ST': 'C3',
          'BIN-SHUFF': 'C5'}

LINE_STYLES = {'original': 'solid',
               'UD': 'solid',
               'UDR': 'solid',
               'ISI-D': 'dashed',
               'JISI-D': 'solid',
               'SHIFT-ST': 'solid',
               'BIN-SHUFF': 'solid'}

# Panel letters
LETTERS = ('A', 'B', 'C', 'D', 'E', 'F')