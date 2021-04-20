import itertools

import numpy as np
import matplotlib.pyplot as plt

import quantities as pq

import yaml

spiketrain_path = '../../data/artificial_data/'
results_path = '../../results/april2021/'

with open("../configfile.yaml", 'r') as stream:
    config = yaml.load(stream, Loader=yaml.Loader)

sep = 2. * config['winlen'] * config['binsize']  # in s
epoch_length = 0.5  # in s

# surr_methods = ['bin_shuffling', 'isi', 'jisi', 'tr_shift', 'ud', 'udrp']
surr_methods = ['ud']

for session, epoch, trialtype, process, surr_method in itertools.product(
        config['sessions'], config['epochs'], config['trialtypes'],
        config['processes'], surr_methods):

    spiketrains = np.load(
        f'{spiketrain_path}{process}/'
        f'{session}/{process}_{epoch}_{trialtype}.npy',
        allow_pickle=True)

    n_trials = int(
        (spiketrains[0].t_stop.simplified.item()
         -spiketrains[0].t_start.simplified.item())/(epoch_length + sep))

    effective_length = n_trials * epoch_length

    rates = np.array([len(spiketrain)/effective_length
             for spiketrain in spiketrains])

    patterns = np.load(
        f'{results_path}{surr_method}/{process}/'
        f'{session}/{epoch}_{trialtype}/filtered_res.npy',
        allow_pickle=True)[0]

    print(patterns)

    neurons = np.unique(
        np.hstack([pattern['neurons'] for pattern in patterns]))
    print(type(neurons[0]))
    # print(rates[np.array([1, 2], dtype=int)])
    print(rates[neurons])

    plt.scatter(rates, np.repeat(1, len(rates)))
    plt.scatter(rates[neurons], np.repeat(1, len(rates[neurons])))
