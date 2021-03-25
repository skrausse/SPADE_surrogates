import os
import sys

import numpy as np
import quantities as pq
import yaml
from yaml import Loader
import rgutils

if __name__ == '__main__':
    with open("configfile.yaml", 'r') as stream:
        config = yaml.load(stream, Loader=Loader)

    sessions = config['sessions']
    epochs = config['epochs']
    trialtypes = config['trialtypes']
    winlen = config['winlen']
    unit = config['unit']
    binsize = (config['binsize'] * pq.s).rescale(unit)
    firing_rate_threshold = config['firing_rate_threshold']
    seed = config['seed']
    SNR_thresh = config['SNR_thresh']
    synchsize = config['synchsize']
    sep = 2 * winlen * binsize

    for session in sessions:
        if not os.path.exists(f'../data/concatenated_spiketrains/{session}'):
            os.makedirs(f'../data/concatenated_spiketrains/{session}')
        for epoch in epochs:
            for trialtype in trialtypes:
                print(f'Loading data {session} {epoch} {trialtype}')
                sts = rgutils.load_epoch_concatenated_trials(
                    session,
                    epoch,
                    trialtypes=trialtype,
                    SNRthresh=SNR_thresh,
                    synchsize=synchsize,
                    sep=sep)
                np.save(f'../data/concatenated_spiketrains/{session}/'
                        f'{epoch}_{trialtype}.npy',
                        sts)

    if firing_rate_threshold is not None:
        sys.path.insert(0, 'analysis_experimental_data')
        import estimate_number_occurrences # create the excluded_neurons file

        excluded_neurons = np.load('excluded_neurons.npy',
                                   allow_pickle=True).item()
        for session in sessions:
            for epoch in epochs:
                for trialtype in trialtypes:
                    sts = list(np.load(f'../data/concatenated_spiketrains/{session}/'
                            f'{epoch}_{trialtype}.npy'))
                    sts = rgutils.filter_neurons(
                        sts, excluded_neurons=excluded_neurons[session])
                    np.save(f'../data/concatenated_spiketrains/{session}/'
                            f'{epoch}_{trialtype}.npy',
                            sts)
