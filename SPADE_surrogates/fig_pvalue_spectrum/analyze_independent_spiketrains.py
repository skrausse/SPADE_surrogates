"""
This module calls the spade run that analyzes the generated data.
"""
import os
import random

import numpy as np

from spade_analyze import SPADE

try:
    from mpi4py import MPI  # for parallelized routines
except ImportError:
    RANK = 0
else:
    comm = MPI.COMM_WORLD  # create MPI communicator
    RANK = comm.Get_rank()  # get rank of current MPI task

surr_methods = ('dither_spikes',
                'dither_spikes_with_refractory_period',
                'joint_isi_dithering',
                'isi_dithering',
                'trial_shifting',
                'bin_shuffling')

if __name__ == '__main__':
    ARRAY_ID = int(os.environ['SLURM_ARRAY_TASK_ID'])
    np.random.seed(RANK)
    random.seed(RANK)
    surr_method = surr_methods[ARRAY_ID]

    setup = SPADE(surr_method=surr_method)

    list_of_spiketrains = np.load(
        file=f'{setup.spiketrain_path}spiketrains_'
             f'{int(setup.rate.magnitude)}_{setup.data_type}.npy',
        allow_pickle=True)

    spade_results = []
    for spiketrains in list_of_spiketrains:
        spade_results.append(setup.run_spade(spiketrains))

    if RANK == 0:
        np.save(
            file=f'{setup.pattern_path}patterns_'
                 f'{int(setup.rate.magnitude)}_'
                 f'{setup.data_type}_{setup.surr_method_short}',
            arr=spade_results)
