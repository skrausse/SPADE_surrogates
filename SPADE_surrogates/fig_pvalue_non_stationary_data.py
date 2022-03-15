import os

from collections import defaultdict
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

import elephant.spade as spade

# config
surr_methods = ('ud', 'udrp', 'jisi', 'isi', 'tr_shift', 'bin_shuffling')
data_types = ('ppd', 'gamma')
sessions = ('i140703-001', 'l101210-001')
epochs = ('start', 'cue1', 'earlydelay', 'latedelay', 'movement', 'hold')
trial_types = ('PGHF', 'PGLF', 'SGHF', 'SGLF')
min_size = 2

LABELS = {'ud': 'UD',
          'udrp': 'UDD',
          'isi': 'ISI-D',
          'jisi': 'JISI-D',
          'tr_shift': 'TR-SHIFT',
          'bin_shuffling': 'WIN-SHUFF'}

# surr_method = surr_methods[0]
data_type = data_types[0]
session = sessions[0]
epoch = epochs[4]
trial_type = trial_types[0]

n_surrogates = 5000
winlen = 12
spectrum = '3d#'

cmap = 'Blues_r'
scatter_size = 45.
color_norm = mcolors.SymLogNorm(
    linthresh=1./n_surrogates,
    vmin=1./n_surrogates,
    vmax=1.,
    base=10.)
centimeters = 1/2.54  # centimeters in inches


def load_results(
        data_type, session, epoch, trial_type,
        surr_method, size):
    path = f'../results/artificial_data/' \
           f'{surr_method}/{data_type}/' \
           f'{session}/{epoch}_{trial_type}'

    job_ids = list(os.walk(path))[0][1]
    job_ids = sorted([int(job_id) for job_id in job_ids])
    job_id = job_ids[size - min_size]

    result_path = f'{path}/{job_id}/results.npy'

    results = np.load(result_path, allow_pickle=True)[0]
    return results


def get_optimized_pvalue_spec(
        data_type, session, epoch, trial_type,
        surr_method, size):

    results = load_results(
        data_type, session, epoch, trial_type,
        surr_method, size)

    pv_spec = results['pvalue_spectrum']

    optimized_pvalue_spec = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int)))

    for entry in pv_spec:
        optimized_pvalue_spec[entry[0]][entry[2]][entry[1]] = entry[3]
    return optimized_pvalue_spec


def plot_pvalue_for_one_size(
        axis, data_type, session, epoch, trial_type,
        surr_method, size, number_of_tests, number_of_tests_per_size,
        max_occ=None, min_occ=None):
    optimized_pvalue_spec = get_optimized_pvalue_spec(
        data_type, session, epoch, trial_type,
        surr_method, size)

    durs = []
    occs = []
    pvs = []

    durs_smaller_thresh = []
    occs_smaller_thresh = []

    durs_smaller_bonf = []
    occs_smaller_bonf = []

    if max_occ is None:
        max_occ = []
        for dur in optimized_pvalue_spec[size].keys():
            for occ in optimized_pvalue_spec[size][dur].keys():
                max_occ.append(occ)

        if len(max_occ) == 0:
            return
        max_occ = max(max_occ)

    for dur in optimized_pvalue_spec[size].keys():
        found_smaller_thresh = False
        found_smaller_bonf = False
        fill = list(
            range(list(optimized_pvalue_spec[size][dur].keys())[-1] + 1,
                  max_occ + 1))
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
            if not found_smaller_bonf and pvalue < 0.05 / number_of_tests:
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
            range(list(optimized_pvalue_spec[size][dur].keys())[-1] + 1,
                  max_occ + 1))
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

    pattern_occ, pattern_dur = get_pattern_occ_duration(
        data_type, session, epoch, trial_type,
        surr_method, size)

    reduced_pattern_occ = []
    reduced_pattern_dur = []

    for occ, dur in zip(pattern_occ, pattern_dur):
        if optimized_pvalue_spec[size][dur][occ] < 0.9999:
            reduced_pattern_occ.append(occ)
            reduced_pattern_dur.append(dur)

    if min_occ is None:
        min_occ = np.min(reduced_pattern_occ)
    axis.set_xlim(left=min_occ-(4/size)**2,
                  right=max_occ+(4/size)**2)

    # axis.scatter(reduced_pattern_occ, reduced_pattern_dur,
    #              s=scatter_size/4, c='C1', marker='x')

    axis.set_yticks(np.arange(1, winlen, 2))
    axis.set_yticklabels(np.arange(2, winlen + 1, 2))

    axis.set_title(
        f'Size {size}, #Tests: {number_of_tests_per_size[size]}')
    return min_occ, max_occ


def create_pvalue_plot(data_type, session, epoch, trial_type, surr_method):
    number_of_tests_per_size, tested_pattern_occs, tested_pattern_durs = \
        get_numbers_of_tests_all_sizes(
            data_type, session, epoch, trial_type, surr_method)
    number_of_tests = sum(number_of_tests_per_size.values())

    fig, axes = plt.subplots(
        2, 2, figsize=(2*8.5*centimeters, 2*10.0*centimeters))
    for axis, size in zip(axes.flatten(), range(2, 6)):
        plot_pvalue_for_one_size(
            axis, data_type, session, epoch, trial_type,
            surr_method, size, number_of_tests, number_of_tests_per_size)

        axis.scatter(tested_pattern_occs[size], tested_pattern_durs[size],
                     s=scatter_size / 4, c='C3', marker='x')

    cbar_map = cm.ScalarMappable(
            norm=color_norm,
            cmap=cmap)

    cbar_map.set_array(np.array([]))
    cbar = fig.colorbar(mappable=cbar_map, ax=axes)

    cbar.set_label('p-value')

    save_path = f'../plots/pvalue_spectra_patterns_spade'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    fig.suptitle(f'{data_type} {session} {epoch} '
                 f'{trial_type} {LABELS[surr_method]} '
                 f'\n #Tests {number_of_tests}')

    fig.savefig(
        f'{save_path}/{data_type}_{session}_{epoch}_{trial_type}_{surr_method}')
    plt.close(fig)


def get_pattern_occ_duration(
        data_type, session, epoch, trial_type,
        surr_method, size):
    results = load_results(
        data_type, session, epoch, trial_type,
        surr_method, size)

    concepts = results['patterns']

    signatures = {(len(concept[0]), len(concept[1]),
                   max(np.array(concept[0]) % winlen))
                  for concept in concepts}
    signatures = np.array(list(signatures))

    pattern_occ = signatures[:, 1]
    pattern_duration = signatures[:, 2]

    return pattern_occ, pattern_duration


def get_number_of_tests(
        data_type, session, epoch, trial_type,
        surr_method, size, verbose=True):
    results = load_results(
        data_type, session, epoch, trial_type,
        surr_method, size)

    pv_spec = results['pvalue_spectrum']
    concepts = results['patterns']

    if verbose:
        print(f'\n{size=}')
        print(f'entries of pvalue-spectrum: {len(pv_spec)}')

    if len(pv_spec) == 0:
        if verbose:
            print(f'number of tests: 0')
        return 0, np.array([]), np.array([])

    pv_spec = np.array(pv_spec)
    mask = spade._mask_pvalue_spectrum(pv_spec, concepts, spectrum, winlen)
    pvalues = pv_spec[:, -1]
    pvalues_totest = pvalues[mask]
    pattern_occ = pv_spec[:, 1][mask]
    pattern_dur = pv_spec[:, 2][mask]

    if verbose:
        print(f'number of tests: {len(pvalues_totest)}')
    return len(pvalues_totest), pattern_occ, pattern_dur


def get_numbers_of_tests_all_sizes(
        data_type, session, epoch, trial_type, surr_method):
    numbers_of_tests = {}
    pattern_occs = {}
    pattern_durs = {}
    for size in range(2, 11):
        number_of_tests, pattern_occ, pattern_dur = get_number_of_tests(
            data_type, session, epoch, trial_type,
            surr_method, size)
        numbers_of_tests[size] = number_of_tests
        pattern_occs[size] = pattern_occ
        pattern_durs[size] = pattern_dur

    return numbers_of_tests, pattern_occs, pattern_durs


def compare_pvalue_spectra(surrogate_methods, sizes):

    fig, axes = plt.subplots(
        nrows=len(surrogate_methods), ncols=len(sizes),
        figsize=(len(sizes)*8*centimeters,
                 len(surrogate_methods)*8*centimeters),
        # sharey='all', sharex='col',
        constrained_layout=True)

    max_occ = {}
    min_occ = {}
    for surr_id, (surr_method, axes_row) in enumerate(
            zip(surrogate_methods[::-1], axes[::-1])):
        print(surr_method)
        number_of_tests_per_size, tested_pattern_occs, tested_pattern_durs = \
            get_numbers_of_tests_all_sizes(
                data_type, session, epoch, trial_type, surr_method)
        number_of_tests = sum(number_of_tests_per_size.values())

        for size, axis in zip(sizes, axes_row):
            if surr_id == 0:
                min_occ[size], max_occ[size] = plot_pvalue_for_one_size(
                    axis, data_type, session, epoch, trial_type,
                    surr_method, size, number_of_tests,
                    number_of_tests_per_size)
            else:
                plot_pvalue_for_one_size(
                    axis, data_type, session, epoch, trial_type,
                    surr_method, size, number_of_tests,
                    number_of_tests_per_size,
                    max_occ=max_occ[size], min_occ=min_occ[size])
            axis.set_title(f'{LABELS[surr_method]}, size {size}')
            axis.set_xlabel('occurrences')
            axis.set_ylabel('duration')

    cbar_map = cm.ScalarMappable(
        norm=color_norm,
        cmap=cmap)

    cbar_map.set_array(np.array([]))
    cbar = fig.colorbar(mappable=cbar_map, ax=axes)

    cbar.set_label('p-value')

    fig.suptitle(f'Comparison of p-value spectra \n'
                 f'{data_type.upper()}, {session}, {epoch}, '
                 f'{trial_type}')
    fig.savefig(
        f'../plots/pvalue_comparison_'
        f'{data_type}_{session}_{epoch}_{trial_type}')

    plt.show()


if __name__ == '__main__':
    pass
    # for data_type, session, epoch, trial_type, surr_method in product(
            #         data_types, sessions, epochs, trial_types, surr_methods):
    # create_pvalue_plot(data_type, session, epoch, trial_type, surr_method)
    # signatures = get_pattern_occ_duration(
    #     data_type, session, epoch, trial_type,
    #     surr_method, size=2)
    # for surr_method in surr_methods:
    #     create_pvalue_plot(data_type, session, epoch, trial_type, surr_method)
    compare_pvalue_spectra(
        surrogate_methods=('ud', 'tr_shift'),
        sizes=(2, 3, 4))
