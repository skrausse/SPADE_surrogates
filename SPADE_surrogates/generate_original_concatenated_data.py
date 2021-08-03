"""
This script loads the experimental spike train data from the 'multielectrode
grasp' folder and saves them as concatenated spike trains.
"""
import os

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
