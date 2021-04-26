import numpy as np
import matplotlib.pyplot as plt

import quantities as pq

from coco import coconad, estpsp, psp2bdr, patred
print('TODO: Add Coconad to the requirements.txt')

from elephant.spike_train_generation import homogeneous_poisson_process as hpp
from elephant.spike_train_surrogates import dither_spikes, surrogates


def get_spiketrains(rate, dead_time, t_stop):
    trains = [hpp(rate=rate, refractory_period=dead_time, t_stop=t_stop) for _ in range(2)]
    trains_mag = [train.magnitude for train in trains]
    return trains, trains_mag


def get_pattern_orig_surr(rate, dead_time, t_stop, surr_methods, dither_parameter, bin_size, n_iter=100):
    bin_size = bin_size.simplified.item()
    patterns = np.zeros(n_iter, dtype=int)
    surrogate_patterns = np.zeros((len(surr_methods), n_iter), dtype=int)

    surr_kwargs = \
        {'bin_shuffling': {'bin_size': bin_size*pq.s},
         'trial_shifting': {'trial_length': 0.5*pq.s,
                            'trial_separation': 0.*pq.s},
         'dither_spikes_with_refractory_period':
             {'refractory_period': bin_size*pq.s},
         'dither_spikes': {},
         'joint_isi_dithering': {},
         'isi_dithering': {}}

    for real_id in range(n_iter):
        trains, trains_mag = get_spiketrains(rate, dead_time, t_stop)

        # Analysis of original spike trains
        pats = coconad(
            trains=zip(range(1, len(trains_mag) + 1), trains_mag),
            supp=10,
            zmin=2,
            zmax=2,
            width=bin_size,
            report="#")

        if len(pats) > 0:
            patterns[real_id] = list(pats.keys())[0][1]

        for surr_id, surr_method in enumerate(surr_methods):
            # Analysis of surrogate spike trains
            dithered_trains = [surrogates(
                train, dt=dither_parameter,
                method=surr_method,
                **surr_kwargs[surr_method])[0] for train in trains]

            dithered_trains_mag = [train.magnitude for train in dithered_trains]

            dithered_pats = coconad(
                trains=zip(range(1, len(dithered_trains_mag) + 1), dithered_trains_mag),
                supp=10,
                zmin=2,
                zmax=2,
                width=bin_size,
                report="#")

            if len(dithered_pats) > 0:
                surrogate_patterns[surr_id, real_id] = list(dithered_pats.keys())[0][1]

    t_stop = t_stop.simplified.item()
    return patterns / t_stop, surrogate_patterns / t_stop


if __name__ == '__main__':
    np.random.seed(0)
    t_stop = 5.*pq.s
    dead_time = 1.6*pq.ms
    rates = np.arange(10., 101., 10.)*pq.Hz
    dither_parameter = 25.*pq.ms
    bin_size = 5.*pq.ms
    surr_methods = ('dither_spikes', 'dither_spikes_with_refractory_period',
                    'joint_isi_dithering', 'isi_dithering', 'bin_shuffling',
                    'trial_shifting')

    mean_patterns = np.empty(shape=rates.shape)
    std_patterns = np.empty(shape=rates.shape)
    mean_surrogate_patterns = np.empty((len(surr_methods), rates.shape[0]))
    std_surrogate_patterns = np.empty((len(surr_methods), rates.shape[0]))

    for rate_id, rate in enumerate(rates):
        patterns, surrogate_patterns = get_pattern_orig_surr(
            rate, dead_time, t_stop, surr_methods, dither_parameter, bin_size)
        mean_patterns[rate_id], std_patterns[rate_id] = np.mean(patterns), np.std(patterns)
        mean_surrogate_patterns[:, rate_id], std_surrogate_patterns[:, rate_id] = \
            np.mean(surrogate_patterns, axis=1), np.std(surrogate_patterns, axis=1)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(rates, mean_patterns)

    for surr_id, surr_method in enumerate(surr_methods):
        ax.plot(rates, mean_surrogate_patterns[surr_id], label=surr_method)
    ax.set_xlabel('firing rate in Hz')
    ax.set_ylabel('Num. Synchronies per sec.')
    ax.legend()
    fig.savefig('../plots/fig_coconad')
