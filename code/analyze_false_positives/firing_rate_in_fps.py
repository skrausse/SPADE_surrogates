import itertools
import os

import numpy as np
import matplotlib.pyplot as plt

import yaml

spiketrain_path = '../../data/artificial_data/'
results_path = '../../results/artificial_data/'

plot_path = '../../plots/fp_firing_distribution/'

if not os.path.exists(plot_path):
    os.mkdir(plot_path)

with open("../configfile.yaml", 'r') as stream:
    config = yaml.load(stream, Loader=yaml.Loader)

sep = 2. * config['winlen'] * config['binsize']  # in s
epoch_length = 0.5  # in s

markersize = 5.

surr_methods = ['ud', 'udrp', 'jisi', 'isi', 'bin_shuffling', 'tr_shift']

excluded_neurons = np.load(
    '../analysis_artificial_data/excluded_neurons.npy',
    allow_pickle=True).item()

contexts = []
for epoch, trialtype in itertools.product(
        config['epochs'], config['trialtypes']):
    if epoch == 'movement':
        epoch = 'mov'
    elif epoch == 'latedelay':
        epoch = 'late'
    elif epoch == 'earlydelay':
        epoch = 'early'
    contexts.append(f'{epoch} {trialtype}')

for session, surr_method in itertools.product(
        config['sessions'], surr_methods):
    fig, axes = plt.subplots(
        nrows=1, ncols=len(config['processes']),
        sharex='all', sharey='all')

    fig.suptitle(f'{session} {surr_method}')

    axes[0].set_yticks(np.arange(len(contexts)))
    axes[0].set_yticklabels(contexts)

    for ax, process in zip(axes, config['processes']):
        ax.set_xlabel(r'$\lambda$ in Hz')
        ax.set_title(process)

        for bev_id, (epoch, trialtype) in enumerate(
                itertools.product(
                    config['epochs'], config['trialtypes'])):

            spiketrains = list(np.load(
                f'{spiketrain_path}{process}/'
                f'{session}/{process}_{epoch}_{trialtype}.npy',
                allow_pickle=True))

            for neuron in excluded_neurons[session]:
                spiketrains.pop(int(neuron))

            n_trials = int(
                (spiketrains[0].t_stop.simplified.item()
                 - spiketrains[0].t_start.simplified.item())/(epoch_length + sep))

            effective_length = n_trials * epoch_length

            rates = np.array(
                [len(spiketrain)/effective_length
                 for spiketrain in spiketrains])

            ax.plot(
                rates, np.repeat(bev_id, len(rates)),
                color='grey', marker='o', linestyle='', alpha=0.3,
                markersize=markersize)

            patterns = np.load(
                f'{results_path}{surr_method}/{process}/'
                f'{session}/{epoch}_{trialtype}/filtered_res.npy',
                allow_pickle=True)[0]

            if len(patterns) == 0:
                continue

            neurons = np.unique(
                np.hstack([pattern['neurons'] for pattern in patterns]))

            ax.plot(
                rates[neurons], np.repeat(bev_id, len(rates[neurons])),
                color='red', marker='o', linestyle='',
                markersize=markersize)
    fig.savefig(
        f'{plot_path}{session}_{surr_method}')
