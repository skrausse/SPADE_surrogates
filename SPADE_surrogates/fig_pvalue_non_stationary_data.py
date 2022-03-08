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

surr_method = surr_methods[0]
data_type = data_types[0]
session = sessions[0]
epoch = epochs[4]
trial_type = trial_types[0]
size = 2

cmap = 'Blues_r'
scatter_size = 45.
color_norm = mcolors.SymLogNorm(
    linthresh=1e-6,
    vmin=1e-6,
    vmax=1.,
    base=10.)

def load_results(
        data_type, session, epoch, trial_type,
        surr_method, size, min_size):
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
        surr_method, size, min_size):

    results = load_results(
        data_type, session, epoch, trial_type,
        surr_method, size, min_size)

    pv_spec = results['pvalue_spectrum']

    optimized_pvalue_spec = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int)))

    for entry in pv_spec:
        optimized_pvalue_spec[entry[0]][entry[2]][entry[1]] = entry[3]
    return optimized_pvalue_spec


def create_pvalue_plot(data_type, session, epoch, trial_type, surr_method):
    centimeters = 1/2.54  # centimeters in inches

    fig, axes = plt.subplots(2, 2, figsize=(2*8.5*centimeters, 2*10.0*centimeters))
    for axis, size in zip(axes.flatten(), range(2, 6)):
        optimized_pvalue_spec = get_optimized_pvalue_spec(
            data_type, session, epoch, trial_type,
            surr_method, size, min_size)

        durs = []
        occs = []
        pvs = []

        durs_smaller_thresh = []
        occs_smaller_thresh = []

        durs_smaller_bonf = []
        occs_smaller_bonf = []

        max_occ = []
        for dur in optimized_pvalue_spec[size].keys():
            for occ in optimized_pvalue_spec[size][dur].keys():
                max_occ.append(occ)

        if len(max_occ) == 0:
            continue
        max_occ = max(max_occ)

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

        axis.set_title(f'Size {size}')

    cbar_map = cm.ScalarMappable(
            norm=color_norm,
            cmap=cmap)

    cbar_map.set_array(np.array([]))
    cbar = fig.colorbar(mappable=cbar_map, ax=axes)

    cbar.set_label('p-value')


    save_path = f'../plots/pvalue_spectra'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    fig.suptitle(f'{data_type} {session} {epoch} {trial_type} {surr_method}')

    fig.savefig(
        f'{save_path}/{data_type}_{session}_{epoch}_{trial_type}_{surr_method}')
    plt.close(fig)


def get_number_of_tests(
        data_type, session, epoch, trial_type,
        surr_method, size, min_size):
    results = load_results(
        data_type, session, epoch, trial_type,
        surr_method, size, min_size)

    pv_spec = results['pvalue_spectrum']
    concepts = results['patterns']
    winlen = 12
    spectrum = '3d#'

    pv_spec = np.array(pv_spec)
    mask = spade._mask_pvalue_spectrum(pv_spec, concepts, spectrum, winlen)
    pvalues = pv_spec[:, -1]

    pvalues_totest = pvalues[mask]
    print(f'{size=}')
    print(f'number of tests: {len(pvalues_totest)}')
    return len(pvalues_totest)


if __name__ == '__main__':
    pass
    # for data_type, session, epoch, trial_type, surr_method in product(
    #         data_types, sessions, epochs, trial_types, surr_methods):
    #     create_pvalue_plot(data_type, session, epoch, trial_type, surr_method)

    for size in range(2, 8):
        try:
            get_number_of_tests(
                data_type, session, epoch, trial_type,
                surr_method, size, min_size)
        except IndexError:
            pass
