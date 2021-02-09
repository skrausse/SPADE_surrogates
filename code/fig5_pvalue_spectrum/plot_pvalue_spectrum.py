"""
This module provides functions to plot the p-value spectra of the SPADE
results.
"""
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

import config as cf


def _load_pv_spectrum(setup):
    results = np.load(
        file=f'{setup.pattern_path}patterns_{int(setup.rate.magnitude)}_'
             f'{setup.data_type}_{setup.surr_method_short}.npy',
        allow_pickle=True)
    return [result['pvalue_spectrum'] for result in results]


def _get_optimized_pvalue_spec(setup):
    optimized_pvalue_spec = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))

    pv_spectrum_list = _load_pv_spectrum(setup)
    for pv_spec in pv_spectrum_list:
        for entry in pv_spec:
            optimized_pvalue_spec[entry[0]][entry[2]][entry[1]].append(
                entry[3])
    return optimized_pvalue_spec


def _get_mean_optimized_pvalue_spec(setup):
    optimized_pvalue_spec = _get_optimized_pvalue_spec(setup)
    for size in optimized_pvalue_spec.keys():
        for dur in optimized_pvalue_spec[size].keys():
            for occ in optimized_pvalue_spec[size][dur].keys():
                optimized_pvalue_spec[size][dur][occ] = sum(
                    optimized_pvalue_spec[size][dur][occ]
                ) / setup.n_realizations
    return optimized_pvalue_spec


def plot_comparison_of_two_pvalue_spectra(setup1, setup2):
    cmap = 'Blues_r'
    scatter_size = 75.

    size = setup1.sizes_to_analyze[0]

    plt.rcParams.update({'font.size': 12,
                         'text.usetex': True})

    color_norm = mcolors.SymLogNorm(
        linthresh=1e-6,
        vmin=1e-6,
        vmax=1.,
        base=10.)

    fig, axes = plt.subplots(2, 1, figsize=(5., 5.), sharex=True)

    # find max_occurrences to fill up with zeroes
    max_occ = []
    for axis_id, (axis, setup) in enumerate(zip(axes, (setup1, setup2))):
        optimized_pvalue_spec = _get_mean_optimized_pvalue_spec(setup)

        for dur in optimized_pvalue_spec[size].keys():
            for occ in optimized_pvalue_spec[size][dur].keys():
                max_occ.append(occ)
    max_occ = max(max_occ)

    for axis_id, (axis, setup) in enumerate(zip(axes, (setup1, setup2))):
        optimized_pvalue_spec = _get_mean_optimized_pvalue_spec(setup)

        durs = []
        occs = []
        pvs = []

        for dur in optimized_pvalue_spec[size].keys():
            fill = list(
                range(list(optimized_pvalue_spec[size][dur].keys())[-1]+1,
                      max_occ+1))
            for occ in list(optimized_pvalue_spec[size][dur].keys()) + fill:
                durs.append(dur)
                occs.append(occ)
                if optimized_pvalue_spec[size][dur][occ]:
                    pvs.append(optimized_pvalue_spec[size][dur][occ])
                else:
                    pvs.append(0)

        durs_zeros = []
        occs_zeros = []
        pvs_zeros = []

        for dur in optimized_pvalue_spec[size].keys():
            fill = list(
                range(list(optimized_pvalue_spec[size][dur].keys())[-1]+1,
                      max_occ+1))
            for occ in list(optimized_pvalue_spec[size][dur].keys()) + fill:
                if not optimized_pvalue_spec[size][dur][occ]:
                    durs_zeros.append(dur)
                    occs_zeros.append(occ)
                    pvs_zeros.append(0)

        axis.scatter(occs, durs, s=scatter_size,
                     c=pvs, marker='s', cmap=cmap,
                     norm=color_norm)

        axis.scatter(occs_zeros, durs_zeros, s=scatter_size,
                     c=pvs_zeros, marker='s',
                     cmap='binary_r')

        if axis_id == 1:
            axis.set_xlabel('number of occurrences')
        axis.set_ylabel('duration')
        if axis_id == 0:
            axis.set_title('Ground Truth')
        else:
            axis.set_title('Uniform Dithering')
        axis.set_yticks(np.arange(1, setup.win_len, 2))
        axis.set_yticklabels(np.arange(2, setup.win_len+1, 2))

    cbar_map = cm.ScalarMappable(
            norm=color_norm,
            cmap=cmap)

    cbar_map.set_array([])
    cbar = fig.colorbar(mappable=cbar_map, ax=axes)

    cbar.set_label('p-value')
    fig.savefig('../../plots/fig5_pvalue_spectrum.eps',
                dpi=300)


if __name__ == '__main__':
    setup_gt = cf.TestCaseSetUp(surr_method='ground_truth')
    setup_ud = cf.TestCaseSetUp(surr_method='dither_spikes')
    plot_comparison_of_two_pvalue_spectra(setup_gt, setup_ud)
