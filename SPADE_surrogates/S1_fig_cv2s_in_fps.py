"""
Script to create S1 Fig that shows the CV2 of the experimental data
(especially for neurons that are part of patterns).
"""
import itertools
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import quantities as pq

import yaml

from generate_artificial_data import get_cv2

spiketrain_path = '../data/artificial_data/'
results_path = '../results/artificial_data/'

plot_path = '../plots/'

if not os.path.exists(plot_path):
    os.mkdir(plot_path)

with open("configfile.yaml", 'r') as stream:
    config = yaml.load(stream, Loader=yaml.Loader)

sep = 2. * config['winlen'] * config['binsize']  # in s
epoch_length = 0.5  # in s

WHAT_TO_PLOT = 'cv2'  # options {'rate', 'cv2'}

markersize = 2.0
figsize = (5.2, 3.2)
alpha = 0.15
rate_limits = (-2., 69.)
rate_ticks = np.arange(0, 61, 10)

legend_loc = 10
yloc_title = 0.95
fontsize_title = 9.

surr_methods = ['ud', 'udrp', 'jisi', 'isi', 'bin_shuffling', 'tr_shift']

excluded_neurons = np.load(
    'analysis_artificial_data/excluded_neurons.npy',
    allow_pickle=True).item()

epoch_labels = []
for epoch in config['epochs']:
    if epoch == 'movement':
        epoch = 'mov'
    elif epoch == 'latedelay':
        epoch = 'late'
    elif epoch == 'earlydelay':
        epoch = 'early'
    epoch_labels.append(epoch)

monkeys = {'i140703-001': 'Monkey N', 'l101210-001': 'Monkey L'}
capitalized_processes = {'ppd': 'PPD', 'gamma': 'Gamma'}


def create_firing_rate_plots(axes, what_to_plot='rate'):
    xlabel = r'$\lambda$ (Hz)' if what_to_plot == 'rate' else 'CV2'
    color_for_combination = {
        'all': 'C0', 'UD': 'C1',
        'UD&\nUDD': 'C2', 'other': 'C3'}

    lines = {'all': None, 'UD': None,
             'UD&\nUDD': None, 'other': None}

    for row_id, (axes_row, session) \
            in enumerate(zip(axes, config['sessions'])):

        for ax, process in zip(axes_row, config['processes']):
            ax.set_yticks(
                (len(config['trialtypes']) + 1) * np.arange(len(epoch_labels)))
            ax.set_yticklabels(epoch_labels)

            if row_id == 1:
                ax.set_xlabel(xlabel, fontsize=fontsize_title)
            if what_to_plot == 'rate':
                ax.set_xlim(rate_limits)
                ax.set_xticks(rate_ticks)
            else:
                ax.set_xlim(0.4, 1.6)
            ax.set_title(
                f'{monkeys[session]} - {capitalized_processes[process]}',
                fontsize=fontsize_title, y=yloc_title)

            for bev_id, ((epoch_id, epoch), trialtype) in enumerate(
                    itertools.product(
                        enumerate(config['epochs']), config['trialtypes'])):

                bev_id = bev_id + epoch_id

                spiketrains = list(np.load(
                    f'{spiketrain_path}{process}/'
                    f'{session}/{process}_{epoch}_{trialtype}.npy',
                    allow_pickle=True))

                for neuron in excluded_neurons[session]:
                    spiketrains.pop(int(neuron))

                if what_to_plot == 'rate':
                    n_trials = int(
                        (spiketrains[0].t_stop.simplified.item()
                         - spiketrains[0].t_start.simplified.item())
                        / (epoch_length + sep))

                    effective_length = n_trials * epoch_length

                    rates = np.array(
                        [len(spiketrain) / effective_length
                         for spiketrain in spiketrains])
                    quantity_to_plot = rates
                else:
                    cv2s = np.array(
                        [get_cv2(spiketrain, sep=sep * pq.s)
                         for spiketrain in spiketrains])
                    quantity_to_plot = cv2s

                ax.plot(
                    quantity_to_plot, np.repeat(bev_id, len(quantity_to_plot)),
                    color='grey', marker='o', linestyle='', alpha=alpha,
                    markersize=markersize)

                neurons_per_method = {}
                all_neurons = np.array([])
                for surr_method in surr_methods:
                    patterns = np.load(
                        f'{results_path}{surr_method}/{process}/'
                        f'{session}/{epoch}_{trialtype}/filtered_res.npy',
                        allow_pickle=True)[0]

                    if len(patterns) == 0:
                        neurons_per_method[surr_method] = np.array([])
                        continue

                    neurons_per_method[surr_method] = np.unique(
                        np.hstack([pattern['neurons']
                                   for pattern in patterns]))
                    all_neurons = np.unique(np.hstack(
                        (all_neurons, neurons_per_method[surr_method])))

                neurons_per_combination = defaultdict(list)
                for neuron in all_neurons:
                    combination = \
                        [surr_method
                         for surr_method in surr_methods
                         if neuron in neurons_per_method[surr_method]]
                    neurons_per_combination[tuple(combination)].append(neuron)

                for combination, neurons in neurons_per_combination.items():
                    neurons = np.array(neurons).astype(int)
                    if len(combination) == len(surr_methods):
                        combination = 'all'
                    elif len(combination) == 1 and combination[0] == 'ud':
                        combination = 'UD'
                    elif len(combination) == 2 and \
                            combination[1] == 'udrp' and \
                            combination[0] == 'ud':
                        combination = 'UD&\nUDD'
                    else:
                        combination = 'other'

                    lines[combination], = ax.plot(
                        quantity_to_plot[neurons],
                        np.repeat(bev_id, len(quantity_to_plot[neurons])),
                        marker='o', linestyle='',
                        markersize=markersize,
                        color=color_for_combination[combination])
    return lines


if __name__ == '__main__':
    fig, axes22 = plt.subplots(
            nrows=len(config['sessions']), ncols=len(config['processes']),
            sharex='all', sharey='all', figsize=figsize)
    plt.subplots_adjust(bottom=0.2, top=0.9, right=0.95)

    legend_lines = create_firing_rate_plots(
        axes22,
        what_to_plot=WHAT_TO_PLOT)

    fig.legend(
        list(legend_lines.values()),
        list(legend_lines.keys()),
        fontsize=fontsize_title,
        fancybox=True, shadow=True, ncol=5, loc="lower left",
        mode="expand", borderaxespad=1)

    plt.show()
    if WHAT_TO_PLOT == 'rate':
        save_name = 'firing_rates_fps'
    else:
        save_name = 'cv2s_fps'

    fig.savefig(f'{plot_path}{save_name}.png')
    # convert manually to eps
    # inkscape cv2s_fps.pdf --export-eps=cv2s_fps.eps
    fig.savefig(f'{plot_path}{save_name}.pdf')
