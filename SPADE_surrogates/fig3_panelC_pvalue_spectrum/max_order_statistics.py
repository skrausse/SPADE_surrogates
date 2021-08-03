"""
This module gives back the expected maxima of ordered binomial statistics.
This are necessary to make an estimate for the min_occ to use within SPADE
analyzing the independent spike trains.
"""

import math

import numpy as np
from scipy.special import binom
import quantities as pq

import config as cf


if __name__ == '__main__':
    max_binomial_statistics = dict()
    np.random.seed(0)

    setup = cf.TestCaseSetUp()

    n = math.floor(
        ((setup.t_stop - setup.t_start) / setup.bin_size).rescale(
            pq.dimensionless).magnitude)
    p_1 = (setup.rate*setup.bin_size).rescale(
        pq.dimensionless).magnitude

    for size in setup.sizes_to_analyze:
        p = p_1**size
        n_combinations = binom(setup.n_spiketrains, size).astype(int)
        max_binomial = np.zeros(shape=100)
        for index in range(100):
            max_binomial[index] = np.max(
                np.random.binomial(n=n, p=p, size=n_combinations))
        mean = np.mean(max_binomial)
        standard_deviation = np.std(max_binomial)

        print(setup.rate, size, mean, standard_deviation)
        max_binomial_statistics[(int(setup.rate), size)] = \
            (mean, standard_deviation)
    np.save(file=f'{setup.pattern_path}max_binomial_statistics.npy',
            arr=max_binomial_statistics)
