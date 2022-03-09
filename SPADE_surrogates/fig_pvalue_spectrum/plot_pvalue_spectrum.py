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
    """
    Function creates the plot comparing to p-value spectra.

    Parameters
    ----------
    setup1: cf.TestCaseSetUp
    setup2: cf.TestCaseSetUp

    """
    cmap = 'Blues_r'
    scatter_size = 45.

    size = 3  # size for the plotting
    lower_limit = 1./(setup1.n_realizations*setup1.n_surrogates)

    color_norm = mcolors.SymLogNorm(
        linthresh=lower_limit,
        vmin=lower_limit,
        vmax=1.,
        base=10.)

    centimeters = 1/2.54  # centimeters in inches
    fig, axes = plt.subplots(  # 12.7
        2, 1, figsize=(8.5*centimeters, 10.0*centimeters), sharex='all',
        gridspec_kw=dict(hspace=0.32, right=0.85, top=0.88, bottom=0.12,
                         left=0.15))

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

        durs_smaller_thresh = []
        occs_smaller_thresh = []

        durs_smaller_bonf = []
        occs_smaller_bonf = []

        for dur in optimized_pvalue_spec[size].keys():
            found_smaller_thresh = False
            found_smaller_bonf = False
            fill = list(
                range(list(optimized_pvalue_spec[size][dur].keys())[-1]+1,
                      max_occ+1))
            for occ in list(optimized_pvalue_spec[size][dur].keys()) + fill:
                durs.append(dur)
                occs.append(occ)
                if optimized_pvalue_spec[size][dur][occ]:
                    pvalue = optimized_pvalue_spec[size][dur][occ]
                else:
                    pvalue = 0
                pvs.append(pvalue)
                if not found_smaller_thresh and pvalue < 0.05:
                    durs_smaller_thresh.append(dur - 0.5)
                    durs_smaller_thresh.append(dur + 0.5)
                    occs_smaller_thresh.append(occ - 0.5)
                    occs_smaller_thresh.append(occ - 0.5)
                    found_smaller_thresh = True
                if not found_smaller_bonf and pvalue < 0.05/(12*4):
                    durs_smaller_bonf.append(dur - 0.5)
                    durs_smaller_bonf.append(dur + 0.5)
                    occs_smaller_bonf.append(occ - 0.5)
                    occs_smaller_bonf.append(occ - 0.5)
                    found_smaller_bonf = True

        # durs_smaller_thresh = np.array(durs_smaller_thresh)
        # occs_smaller_thresh = np.array(occs_smaller_thresh)

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

        axis.plot(occs_smaller_thresh, durs_smaller_thresh)

        axis.plot(occs_smaller_bonf, durs_smaller_bonf)

        if axis_id == 1:
            axis.set_xlabel('number of occurrences')
        axis.set_ylabel('duration', labelpad=1.8)
        if axis_id == 0:
            axis.set_title(f'{setup1.surr_method_short}')
        else:
            axis.set_title(f'{setup2.surr_method_short}')
        axis.set_yticks(np.arange(1, setup.win_len, 2))
        axis.set_yticklabels(np.arange(2, setup.win_len+1, 2))

    cbar_map = cm.ScalarMappable(
            norm=color_norm,
            cmap=cmap)

    cbar_map.set_array(np.array([]))
    cbar = fig.colorbar(mappable=cbar_map, ax=axes)

    cbar.set_label('p-value')
    fig.suptitle(f'P-value spectrum (PPD; d={setup1.dead_time.item()}ms)')
    fig.savefig(f'../../plots/fig3_panelC_pvalue_spectrum_{setup2.surr_method_short}.svg',
                dpi=300)
    fig.savefig(f'../../plots/fig3_panelC_pvalue_spectrum_{setup2.surr_method_short}.png',
                dpi=300)
    plt.show()


def plot_flattened_pvalue_spectra(setup1, setup2):
    """
    Function creates a plot comparing to p-value spectra, showing the p-values
    as function of each other.

    Parameters
    ----------
    setup1: cf.TestCaseSetUp
    setup2: cf.TestCaseSetUp

    """

    plt.rcParams.update({'font.size': 12,
                         'text.usetex': True})

    fig, ax = plt.subplots(1, 1, figsize=(5., 5.))

    optimized_pvalue_spec1 = _get_mean_optimized_pvalue_spec(setup1)
    optimized_pvalue_spec2 = _get_mean_optimized_pvalue_spec(setup2)

    pvalues1 = []
    pvalues2 = []

    for size in optimized_pvalue_spec1.keys():
        for dur in optimized_pvalue_spec1[size].keys():
            for occ in optimized_pvalue_spec1[size][dur].keys():
                pvalues1.append(optimized_pvalue_spec1[size][dur][occ])
                if isinstance(optimized_pvalue_spec2[size][dur][occ], float):
                    pvalues2.append(optimized_pvalue_spec2[size][dur][occ])
                else:
                    pvalues2.append(0.)

    ax.scatter(np.array(pvalues1), np.array(pvalues2))
    ax.set_xlabel('p-values GT')
    ax.set_ylabel('p-values UD')


if __name__ == '__main__':
    setup_ud = cf.TestCaseSetUp(surr_method='dither_spikes')
    for surr_method in ('dither_spikes_with_refractory_period',
                        'joint_isi_dithering',
                        'isi_dithering',
                        'trial_shifting',
                        'bin_shuffling'):
        setup_2 = cf.TestCaseSetUp(surr_method=surr_method)
        plot_comparison_of_two_pvalue_spectra(setup_ud, setup_2)
