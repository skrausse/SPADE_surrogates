"""
Script to create S2 Fig that shows the difference between CoCoNAD and FIM.
"""
import numpy as np
import matplotlib.pyplot as plt

import quantities as pq

from elephant import spade
from elephant.spike_train_generation import homogeneous_poisson_process as hpp
from elephant.spike_train_surrogates import surrogates

from coco import coconad
print('TODO: Add Coconad to the requirements.txt')


surr_methods = ('dither_spikes', 'dither_spikes_with_refractory_period',
                'joint_isi_dithering', 'isi_dithering', 'bin_shuffling',
                'trial_shifting')

LABELS = {'original': 'original',
          'dither_spikes': 'UD',
          'dither_spikes_with_refractory_period': 'UDD',
          'isi_dithering': 'ISI-D',
          'joint_isi_dithering': 'JISI-D',
          'trial_shifting': 'TR-SHIFT',
          'bin_shuffling': 'WIN-SHUFF'}

COLORS = {'original': 'C0',
          'dither_spikes': 'C1',
          'dither_spikes_with_refractory_period': 'C2',
          'isi_dithering': 'C4',
          'joint_isi_dithering': 'C6',
          'trial_shifting': 'C3',
          'bin_shuffling': 'C5'}


def _get_spiketrains(rate, dead_time, t_stop):
    trains = [hpp(
        rate=rate, refractory_period=dead_time, t_stop=t_stop)
        for _ in range(2)]
    trains_mag = [train.magnitude for train in trains]
    return trains, trains_mag


def _get_pattern_orig_surr(
        rate, dead_time, t_stop, surr_methods, dither_parameter,
        bin_size, n_iter=100):
    coconad_bin_size = bin_size.simplified.item()/2
    patterns_coco = np.zeros(n_iter, dtype=int)
    patterns_fim = np.zeros(n_iter, dtype=int)
    surrogate_patterns_coco = np.zeros((len(surr_methods), n_iter), dtype=int)
    surrogate_patterns_fim = np.zeros((len(surr_methods), n_iter), dtype=int)

    surr_kwargs = \
        {'bin_shuffling': {'bin_size': bin_size},
         'trial_shifting': {'trial_length': 0.5*pq.s,
                            'trial_separation': 0.*pq.s},
         'dither_spikes_with_refractory_period':
             {'refractory_period': bin_size},
         'dither_spikes': {},
         'joint_isi_dithering': {},
         'isi_dithering': {}}

    for real_id in range(n_iter):
        trains, trains_mag = _get_spiketrains(rate, dead_time, t_stop)

        # Analysis of original spike trains with coconad
        pats_coco = coconad(
            trains=zip(range(1, len(trains_mag) + 1), trains_mag),
            supp=10,
            zmin=2,
            zmax=2,
            width=coconad_bin_size,
            report="#")

        if len(pats_coco) > 0:
            patterns_coco[real_id] = list(pats_coco.keys())[0][1]

        # Analysis of original spike trains with fpgrowth
        pats_fim = spade.spade(spiketrains=trains,
                               bin_size=bin_size,
                               winlen=1,
                               min_spikes=2,
                               max_spikes=2,
                               min_occ=10)

        if len(pats_fim['patterns']) > 0:
            patterns_fim[real_id] = pats_fim['patterns'][0]['signature'][1]

        for surr_id, surr_method in enumerate(surr_methods):
            # Analysis of surrogate spike trains
            dithered_trains = [surrogates(
                train, dt=dither_parameter,
                method=surr_method,
                **surr_kwargs[surr_method])[0] for train in trains]

            dithered_trains_mag = [train.magnitude
                                   for train in dithered_trains]

            dithered_pats_coco = coconad(
                trains=zip(range(1, len(dithered_trains_mag) + 1),
                           dithered_trains_mag),
                supp=10,
                zmin=2,
                zmax=2,
                width=coconad_bin_size,
                report="#")

            if len(dithered_pats_coco) > 0:
                surrogate_patterns_coco[surr_id, real_id] =\
                    list(dithered_pats_coco.keys())[0][1]

            dithered_pats_fim = spade.spade(
                spiketrains=dithered_trains,
                bin_size=bin_size,
                winlen=1,
                min_spikes=2,
                max_spikes=2,
                min_occ=10)

            if len(dithered_pats_fim['patterns']) > 0:
                surrogate_patterns_fim[surr_id, real_id] =\
                    dithered_pats_fim['patterns'][0]['signature'][1]

    t_stop = t_stop.simplified.item()
    return (patterns_coco / t_stop, surrogate_patterns_coco / t_stop,
            patterns_fim / t_stop, surrogate_patterns_fim / t_stop)


if __name__ == '__main__':
    np.random.seed(0)
    t_stop = 5.*pq.s
    dead_time = 1.6*pq.ms
    rates = np.arange(10., 101., 10.)*pq.Hz
    dither_parameter = 25.*pq.ms
    bin_size = 5.*pq.ms  # so 2.5 ms for CoCoNad
    surr_methods = ('dither_spikes', 'dither_spikes_with_refractory_period',
                    'joint_isi_dithering', 'isi_dithering', 'bin_shuffling',
                    'trial_shifting')

    mean_patterns_coco = np.empty(shape=rates.shape)
    std_patterns_coco = np.empty(shape=rates.shape)
    mean_surrogate_patterns_coco = np.empty(
        (len(surr_methods), rates.shape[0]))
    std_surrogate_patterns_coco = np.empty((len(surr_methods), rates.shape[0]))
    mean_patterns_fim = np.empty(shape=rates.shape)
    std_patterns_fim = np.empty(shape=rates.shape)
    mean_surrogate_patterns_fim = np.empty((len(surr_methods), rates.shape[0]))
    std_surrogate_patterns_fim = np.empty((len(surr_methods), rates.shape[0]))

    for rate_id, rate in enumerate(rates):
        (patterns_coco, surrogate_patterns_coco,
         patterns_fim, surrogate_patterns_fim) = _get_pattern_orig_surr(
            rate, dead_time, t_stop, surr_methods, dither_parameter, bin_size)
        mean_patterns_coco[rate_id] = np.mean(patterns_coco)
        std_patterns_coco[rate_id] = np.std(patterns_coco)

        mean_surrogate_patterns_coco[:, rate_id] = \
            np.mean(surrogate_patterns_coco, axis=1)
        std_surrogate_patterns_coco[:, rate_id] = \
            np.std(surrogate_patterns_coco, axis=1)

        mean_patterns_fim[rate_id] = np.mean(patterns_fim)
        std_patterns_fim[rate_id] = np.std(patterns_fim)

        mean_surrogate_patterns_fim[:, rate_id] = \
            np.mean(surrogate_patterns_fim, axis=1)
        std_surrogate_patterns_fim[:, rate_id] = \
            np.std(surrogate_patterns_fim, axis=1)

    fig, ax = plt.subplots(1, 2, figsize=(6, 3), sharey='all', dpi=300)
    print('CoCoNAD')
    print('rates', rates)
    print('original', mean_patterns_coco)
    ax[0].plot(rates, mean_patterns_coco, label='original',
               color=COLORS['original'], linewidth=2.)
    for surr_id, surr_method in enumerate(surr_methods):
        print(surr_method, mean_surrogate_patterns_coco[surr_id])
        ax[0].plot(rates, mean_surrogate_patterns_coco[surr_id],
                   label=LABELS[surr_method], color=COLORS[surr_method])
    ax[0].set_ylabel('Num. Synchronies per sec.')
    ax[0].set_xlabel('firing rate in Hz', labelpad=-2.)
    ax[0].set_title('CoCoNAD')
    # ax[0].legend()
    print('FIM')
    print('rates', rates)
    print('original', mean_patterns_fim)
    ax[1].plot(rates, mean_patterns_fim, label='original',
               color=COLORS['original'], linewidth=2.)
    for surr_id, surr_method in enumerate(surr_methods):
        print(surr_method, mean_surrogate_patterns_fim[surr_id])
        ax[1].plot(rates, mean_surrogate_patterns_fim[surr_id],
                   label=LABELS[surr_method], color=COLORS[surr_method])
    ax[1].set_xlabel('firing rate in Hz', labelpad=-2.)
    ax[1].set_title('FIM')
    ax[1].legend(fontsize='x-small')
    # ax[1].set_ylabel('Num. Synchronies per sec.')
    fig.savefig('../plots/fig_coconad.png')
    fig.savefig('../plots/fig_coconad.eps')
    plt.show()
