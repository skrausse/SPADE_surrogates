import os

from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

# config
surr_methods = ('ud', 'udrp', 'jisi', 'isi', 'tr_shift', 'bin_shuffling')
data_types = ('ppd', 'gamma')
sessions = ('i140703-001', 'l101210-001')
epochs = ('start', 'cue1', 'earlydelay', 'latedelay', 'movement', 'hold')
trial_types = ('PGHF', 'PGLF', 'SGHF', 'SGLF')
min_size = 2

surr_method = surr_methods[0]
data_type = data_types[0]
session = sessions[0]
epoch = epochs[4]
trial_type = trial_types[0]
size = 3

path = f'../results/artificial_data/' \
       f'{surr_method}/{data_type}/' \
       f'{session}/{epoch}_{trial_type}'

job_ids = list(os.walk(path))[0][1]
job_ids = sorted([int(job_id) for job_id in job_ids])
job_id = job_ids[size-min_size]

result_path = f'{path}/{job_id}/results.npy'


def get_optimized_pvalue_spec(result_path):
    optimized_pvalue_spec = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))

    results = np.load(result_path, allow_pickle=True)[0]

    pv_spec = results['pvalue_spectrum']

    for entry in pv_spec:
        optimized_pvalue_spec[entry[0]][entry[2]][entry[1]].append(
                entry[3])
    return optimized_pvalue_spec

fig, axis = plt.subplots()
optimized_pvalue_spec = get_optimized_pvalue_spec(result_path)

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