"""
In this script are all functions gathered for a numerical assessment of
 statistical quantities after applying a surrogate method.
"""
import time
import random

from collections import defaultdict

import numpy as np
import quantities as pq
import neo

from elephant import conversion as conv
from elephant import spike_train_correlation as corr
from elephant import statistics as stat
from elephant import spike_train_generation as stg
from elephant import spike_train_surrogates as surr

import fig7_surrogate_statistics_config as cfg

DATA_PATH = cfg.DATA_PATH

SURR_METHODS = cfg.SURR_METHODS
DATA_TYPES = cfg.DATA_TYPES

DITHER = cfg.DITHER
DEAD_TIME = cfg.DEAD_TIME
SHAPE_FACTOR = cfg.SHAPE_FACTOR


def _create_spiketrain(data_type, rate, t_start, t_stop):
    if data_type == 'Poisson':
        spiketrain = stg.homogeneous_poisson_process(
            rate=rate, t_start=t_start, t_stop=t_stop)
    elif data_type == 'PPD':
        spiketrain = stg.homogeneous_poisson_process(
            rate=rate, t_start=t_start, t_stop=t_stop,
            refractory_period=DEAD_TIME)
    elif data_type == 'Gamma':
        spiketrain = stg.homogeneous_gamma_process(
            a=SHAPE_FACTOR, b=SHAPE_FACTOR * rate, t_start=t_start,
            t_stop=t_stop)
    else:
        raise ValueError('data_type should be one of Poisson, PPD, or Gamma')
    return spiketrain


def _get_dithered_spiketrains(spiketrain, surr_method, dither, spade_bin_size,
                              n_surrogates):
    if surr_method == 'UD':
        dithered_spiketrains = \
            surr.dither_spikes(
                spiketrain=spiketrain,
                dither=dither, n_surrogates=n_surrogates)
    elif surr_method == 'UDR':
        dithered_spiketrains = \
            surr.dither_spikes(
                spiketrain=spiketrain,
                n_surrogates=n_surrogates,
                dither=dither,
                refractory_period=10 * pq.ms)
    elif surr_method == 'JISI-D':
        dithered_spiketrains = surr.JointISI(
            spiketrain=spiketrain, dither=dither, method='window',
            cutoff=False, refractory_period=0., sigma=0.
        ).dithering(n_surrogates=n_surrogates)
    elif surr_method == 'ISI-D':
        dithered_spiketrains = surr.JointISI(
            spiketrain=spiketrain, dither=dither, method='window',
            isi_dithering=True,
            cutoff=False, refractory_period=0., sigma=0.
        ).dithering(n_surrogates=n_surrogates)
    elif surr_method == 'SHIFT-ST':
        dithered_spiketrains = surr.surrogates(
            method='trial_shifting',
            spiketrain=spiketrain, dt=dither, trial_length=500. * pq.ms,
            trial_separation=0. * pq.ms, n_surrogates=n_surrogates
        )
    elif surr_method == 'BIN-SHUFF':
        dithered_spiketrains = surr.surrogates(
            method='bin_shuffling',
            spiketrain=spiketrain,
            dt=dither, bin_size=spade_bin_size,
            n_surrogates=n_surrogates
        )
    else:
        raise ValueError('surr_method is unknown')
    return dithered_spiketrains


def _isi(spiketrain, rate, bin_size, data_type, surr_method):
    isi = np.diff(spiketrain.rescale(pq.ms).magnitude)
    hist, bin_edges = np.histogram(
        a=isi,
        bins=np.arange(0., 3 * (1. / rate).rescale(pq.ms).magnitude,
                       bin_size.rescale(pq.ms).magnitude)
    )

    hist = hist * 1000. / len(isi)

    results = {'hist': hist,
               'bin_edges': bin_edges}

    np.save(f'{DATA_PATH}/isi_{data_type}_{surr_method}.npy', results)


def _ac_cc(spiketrain, bin_size, num_bins,
           rate, t_stop, data_type, surr_method, spiketrain2=None):
    binned_st = conv.BinnedSpikeTrain(spiketrain, binsize=bin_size)
    ac_hist = corr.cross_correlation_histogram(
        binned_st1=binned_st,
        binned_st2=binned_st,
        window=[-num_bins, num_bins],
        border_correction=False,
        binary=False,
        kernel=None,
        method='speed'
    )[0]
    # ac_hist[num_bins, 0] -= len(spiketrain)

    ac_hist_times = ac_hist.times.rescale(pq.ms).magnitude\
        + ac_hist.sampling_period.rescale(pq.ms).magnitude / 2
    ac_hist = ac_hist[:, 0].magnitude / \
        (rate * bin_size * t_stop).simplified.magnitude

    results = {'hist_times': ac_hist_times,
               'hist': ac_hist}

    np.save(f'{DATA_PATH}/ac_{data_type}_{surr_method}.npy',
            results)

    if spiketrain2 is None:
        np.save(f'{DATA_PATH}/cc_{data_type}_{surr_method}.npy',
                results)
        return

    binned_st2 = conv.BinnedSpikeTrain(
        spiketrain2, binsize=bin_size)

    cc_cross = corr.cross_correlation_histogram(
        binned_st1=binned_st,
        binned_st2=binned_st2,
        window=[-num_bins, num_bins],
        border_correction=False,
        binary=False,
        kernel=None,
        method='speed'
    )[0]

    cc_hist_times = cc_cross.times.rescale(pq.ms).magnitude \
        + cc_cross.sampling_period.rescale(pq.ms).magnitude / 2

    cc_hist = cc_cross[:, 0].magnitude / \
        (rate * bin_size * t_stop).simplified.magnitude

    results = {'hist_times': cc_hist_times,
               'hist': cc_hist}

    np.save(f'{DATA_PATH}/cc_{data_type}_{surr_method}.npy', results)


def _rate_eff_moved_indep(
        spiketrain, spiketrain2, clipped_rates, data_type, t_stop,
        spade_bin_size, eff_moved):
    clipped_rates[data_type] = \
        {'rate': (len(spiketrain) / t_stop).rescale(pq.Hz)}

    binned_st_bool = conv.BinnedSpikeTrain(
        spiketrain, binsize=spade_bin_size).to_bool_array()
    binned_st = conv.BinnedSpikeTrain(
        spiketrain, binsize=spade_bin_size).to_array()
    binned_st_2 = conv.BinnedSpikeTrain(
        spiketrain2, binsize=spade_bin_size).to_array()

    clipped_rates[data_type]['binned'] = \
        np.sum(binned_st_bool) / len(spiketrain)

    eff_moved[data_type] = dict()
    eff_moved[data_type]['indep.'] = \
        np.sum(np.abs(binned_st - binned_st_2)) / 2 \
        / np.sqrt(len(spiketrain) * len(spiketrain2))


def _rate_eff_moved(
        spiketrain, dithered_spiketrain, spade_bin_size,
        clipped_rates, data_type, surr_method, eff_moved):

    binned_st_bool = conv.BinnedSpikeTrain(
        spiketrain, bin_size=spade_bin_size).to_bool_array()
    binned_st = conv.BinnedSpikeTrain(
        spiketrain, bin_size=spade_bin_size).to_array()

    binned_dithered_st = conv.BinnedSpikeTrain(
        dithered_spiketrain, binsize=spade_bin_size).to_bool_array()

    clipped_rates[data_type][surr_method] = \
        np.sum(binned_dithered_st) / len(spiketrain)

    eff_moved[data_type][surr_method] =\
        np.sum(np.abs(binned_st - binned_dithered_st)) / 2 \
        / np.sqrt(len(spiketrain) * len(dithered_spiketrain))


def _statistical_analysis_of_single_rate(rate=10 * pq.Hz,
                                         num_spikes=10000):
    clipped_rates = {}
    eff_moved = {}

    t_start = 0. * pq.s
    t_stop = (num_spikes / rate).rescale(pq.s)
    spade_bin_size = 5. * pq.ms

    for type_id, data_type in enumerate(DATA_TYPES):
        np.random.seed(1)
        random.seed(1)

        spiketrain = _create_spiketrain(data_type, rate, t_start, t_stop)

        spiketrain2 = _create_spiketrain(data_type, rate, t_start, t_stop)

        _rate_eff_moved_indep(
            spiketrain, spiketrain2, clipped_rates, data_type,  t_stop,
            spade_bin_size, eff_moved)

        for surr_method in SURR_METHODS:
            np.random.seed(0)
            random.seed(0)

            dithered_spiketrain = \
                _get_dithered_spiketrains(spiketrain, surr_method, DITHER,
                                          spade_bin_size, n_surrogates=1)[0]

            _rate_eff_moved(
                spiketrain, dithered_spiketrain, spade_bin_size,
                clipped_rates, data_type, surr_method, eff_moved)
    return clipped_rates, eff_moved


def statistics_overview():
    clipped_firing_rate_and_eff_moved()

    statistical_analysis_of_single_rate(
        num_spikes=100000)


def statistical_analysis_of_single_rate(
        rate=50 * pq.Hz, num_spikes=500000):

    t_start = 0. * pq.s
    t_stop = (num_spikes / rate).rescale(pq.s)
    bin_size = 1. * pq.ms
    spade_bin_size = 5. * pq.ms
    num_bins = 60

    for type_id, data_type in enumerate(DATA_TYPES):
        np.random.seed(1)
        random.seed(1)

        time_start = time.time()
        spiketrain = _create_spiketrain(data_type, rate, t_start, t_stop)

        _isi(spiketrain, rate, bin_size, data_type, surr_method='original')

        _ac_cc(spiketrain, bin_size, num_bins,
               rate, t_stop, data_type, surr_method='original')
        time_stop = time.time()
        print('orig.', data_type, time_stop - time_start)

        for surr_method in SURR_METHODS:
            np.random.seed(0)
            random.seed(0)

            time_start = time.time()
            dithered_spiketrain, dithered_spiketrain2 = \
                _get_dithered_spiketrains(spiketrain, surr_method, DITHER,
                                          spade_bin_size, n_surrogates=2)

            _isi(dithered_spiketrain, rate, bin_size, data_type, surr_method)

            _ac_cc(dithered_spiketrain, bin_size, num_bins,
                   rate, t_stop, data_type, surr_method,
                   spiketrain2=dithered_spiketrain2)
            time_stop = time.time()
            print(surr_method, data_type, time_stop-time_start)


def _convert_nested_defaultdict(default_dict_object):
    new_dict_object = {}
    for key in default_dict_object.keys():
        new_dict_object[key] = {}
        for key2 in default_dict_object[key].keys():
            new_dict_object[key][key2] = default_dict_object[key][key2]
    return new_dict_object


def clipped_firing_rate_and_eff_moved():
    """
    This function makes a plot for the clipped firing rate,
    the ratio of spike in same bins and the ratio
    of identical spikes.
    Spike train types are PPR and Gamma. The surrogate methods are uniform
    dithering, UDR and joint-ISI dithering.

    Returns
    -------
    None
    """
    initial_rates = np.arange(10., 100.1, 10.) * pq.Hz

    rates = defaultdict(list)
    ratio_clipped = defaultdict(list)
    ratio_clipped_surr = defaultdict(lambda: defaultdict(list))

    ratio_indep_moved = defaultdict(list)
    ratio_moved = defaultdict(lambda: defaultdict(list))

    for rate in initial_rates:
        clipped_rates, eff_moved = _statistical_analysis_of_single_rate(rate)
        for type_id, data_type in enumerate(DATA_TYPES):
            rates[type_id].append(clipped_rates[data_type]['rate'])
            ratio_clipped[type_id].append(clipped_rates[data_type]['binned'])
            ratio_indep_moved[type_id].append(eff_moved[data_type]['indep.'])
            for surr_method in SURR_METHODS:
                ratio_clipped_surr[type_id][surr_method].append(
                    clipped_rates[data_type][surr_method])
                ratio_moved[type_id][surr_method].append(
                    eff_moved[data_type][surr_method])

    results = {'rates': dict(rates),
               'ratio_clipped': dict(ratio_clipped),
               'ratio_clipped_surr': _convert_nested_defaultdict(
                   ratio_clipped_surr),
               'ratio_indep_moved': dict(ratio_indep_moved),
               'ratio_moved': _convert_nested_defaultdict(ratio_moved)}

    np.save(f'{DATA_PATH}/clipped_rates.npy', results)


def firing_rate_change():
    """
    This function creates a plot which shows the change in firing rate profile
    after applying a surrogate method. Starting point is a process (either PPR
    or Gamma) with that has the first 50 ms a firing rate of 10 Hz and than of
    80 Hz, chosen similarly to Louis et al. (2010).
    The surrogates are than created with uniform dithering, UDR and joint-ISI
    dithering.
    The plot is saved in the plots folder.

    Returns
    -------
    None

    """

    rate_1 = 10 * pq.Hz
    rate_2 = 80 * pq.Hz
    t_start = 0. * pq.ms
    t_stop = 150. * pq.ms
    units = t_stop.units
    t_stop_mag = t_stop.magnitude

    dither_mag = DITHER.magnitude
    number_of_spiketrains = 10000
    # number_of_spiketrains = 500

    bin_size = 1. * pq.ms
    spade_bin_size = 5. * pq.ms

    results = {}

    time_start = time.time()
    for type_id, data_type in enumerate(DATA_TYPES):
        results[data_type] = {}
        np.random.seed(1)
        spiketrains = []
        for _ in range(number_of_spiketrains):
            if data_type == 'Poisson':
                spiketrain_1 = stg.homogeneous_poisson_process(
                    rate=rate_1, t_start=t_start - DITHER,
                    t_stop=t_stop + DITHER)
                spiketrain_2 = stg.homogeneous_poisson_process(
                    rate=rate_2, t_start=t_start - DITHER,
                    t_stop=t_stop + DITHER)
            elif data_type == 'PPD':
                spiketrain_1 = stg.homogeneous_poisson_process(
                    rate=rate_1, t_start=t_start-DITHER, t_stop=t_stop+DITHER,
                    refractory_period=DEAD_TIME)
                spiketrain_2 = stg.homogeneous_poisson_process(
                    rate=rate_2, t_start=t_start-DITHER, t_stop=t_stop+DITHER,
                    refractory_period=DEAD_TIME)
            elif data_type == 'Gamma':
                spiketrain_1 = stg.homogeneous_gamma_process(
                    a=SHAPE_FACTOR,
                    b=SHAPE_FACTOR * rate_1,
                    t_start=t_start - DITHER,
                    t_stop=t_stop + DITHER)
                spiketrain_2 = stg.homogeneous_gamma_process(
                    a=SHAPE_FACTOR,
                    b=SHAPE_FACTOR * rate_2,
                    t_start=t_start - DITHER,
                    t_stop=t_stop + DITHER)
            else:
                raise ValueError('data_type should be Poisson, PPD, or Gamma')
            spiketrain_1 = spiketrain_1.magnitude
            spiketrain_2 = spiketrain_2.magnitude
            spiketrain = neo.SpikeTrain(
                times=np.hstack(
                    (spiketrain_1[spiketrain_1 > t_stop_mag/2]
                     - t_stop_mag / 2 - dither_mag,
                     spiketrain_2[spiketrain_2 > t_stop_mag/2])
                ),
                t_start=t_start-DITHER,
                t_stop=t_stop+DITHER,
                units=units)
            spiketrains.append(spiketrain)

        rate_original = stat.time_histogram(spiketrains, binsize=bin_size,
                                            t_start=t_start, t_stop=t_stop,
                                            output='rate')
        results[data_type]['original'] = rate_original
        print('orig.', time.time()-time_start)

        for surr_method in SURR_METHODS:
            time_start = time.time()
            np.random.seed(0)
            if surr_method == 'UD':
                dithered_spiketrains = [
                    surr.dither_spikes(
                        spiketrain=spiketrain, edges=False,
                        dither=DITHER)[0]
                    for spiketrain in spiketrains]
            elif surr_method == 'UDR':
                dithered_spiketrains = [
                    surr.dither_spikes(
                        spiketrain=spiketrain,
                        dither=DITHER,
                        refractory_period=10 * pq.ms)[0]
                    for spiketrain in spiketrains]
            elif surr_method in ('ISI-D', 'JISI-D'):
                concatenated_spiketrain = neo.SpikeTrain(
                    times=np.hstack([
                        spiketrain.magnitude + dither_mag
                        + st_id * (t_stop_mag + 2*dither_mag)
                        for st_id, spiketrain in enumerate(spiketrains)]),
                    t_start=t_start,
                    t_stop=len(spiketrains) * (t_stop + 2 * DITHER),
                    units=units
                )
                if surr_method == 'JISI-D':
                    dithered_conc_spiketrain = surr.JointISI(
                        spiketrain=concatenated_spiketrain, dither=DITHER,
                        method='window',
                        cutoff=False, refractory_period=0., sigma=0.
                        ).dithering()[0]
                else:
                    dithered_conc_spiketrain = surr.JointISI(
                        spiketrain=concatenated_spiketrain, dither=DITHER,
                        method='window', isi_dithering=True,
                        cutoff=False, refractory_period=0., sigma=0.
                    ).dithering()[0]
                dithered_conc_mag = dithered_conc_spiketrain.magnitude

                dithered_spiketrains = \
                    [neo.SpikeTrain(
                        times=dithered_conc_mag[np.all(
                            (dithered_conc_mag > st_id*(
                                t_stop_mag+2*dither_mag),
                             dithered_conc_mag < (st_id+1)*(
                                 t_stop_mag+2*dither_mag)), axis=0)]
                        - st_id*(t_stop_mag+2*dither_mag) - dither_mag,
                        t_start=-DITHER,
                        t_stop=t_stop+DITHER,
                        units=units)
                     for st_id in range(len(spiketrains))]
            elif surr_method == 'SHIFT-ST':
                trial_length = spiketrains[0].t_stop - spiketrains[0].t_start
                dithered_spiketrains = [surr.surrogates(
                    method='trial_shifting',
                    spiketrain=spiketrain, dt=DITHER,
                    trial_length=trial_length,
                    trial_separation=0. * pq.ms)[0]
                    for spiketrain in spiketrains
                ]
            elif surr_method == 'BIN-SHUFF':
                dithered_spiketrains = []
                for spiketrain in spiketrains:
                    spiketrain = spiketrain.magnitude
                    spiketrain = spiketrain[
                        np.all((spiketrain > 0., spiketrain < t_stop_mag),
                               axis=0)]
                    spiketrain = neo.SpikeTrain(spiketrain, t_stop=t_stop,
                                                t_start=t_start, units=units)
                    dithered_spiketrain = surr.surrogates(
                        method='bin_shuffling',
                        spiketrain=spiketrain,
                        dt=DITHER, bin_size=spade_bin_size)[0]
                    dithered_spiketrains.append(dithered_spiketrain)
            else:
                raise ValueError('surr_method unknown')

            rate_dithered = stat.time_histogram(dithered_spiketrains,
                                                binsize=bin_size,
                                                t_start=t_start, t_stop=t_stop,
                                                output='rate')
            print(surr_method, time.time()-time_start)

            results[data_type][surr_method] = rate_dithered

        np.save(f'{DATA_PATH}/rate_step.npy', results)


def cv_change():
    results = {}
    num_spikes = 10000
    rate = 50*pq.Hz
    t_start = 0. * pq.s
    t_stop = (num_spikes / rate).rescale(pq.s)
    spade_bin_size = 5. * pq.ms

    cvs = np.arange(0.4, 1.21, 0.05)
    shape_factors = 1./cvs**2

    np.random.seed(0)
    spiketrains = [stg.homogeneous_gamma_process(
        a=shape_factor, b=rate*shape_factor, t_start=t_start, t_stop=t_stop)
        for shape_factor in shape_factors]
    cvs_real = [stat.cv(np.diff(spiketrain)) for spiketrain in spiketrains]
    results['cvs_real'] = cvs_real

    random.seed(1)
    np.random.seed(1)
    for surr_method in SURR_METHODS:
        dithered_spiketrains = \
            [_get_dithered_spiketrains(
                spiketrain, surr_method, DITHER, spade_bin_size,
                n_surrogates=1)[0] for spiketrain in spiketrains]
        cvs_dithered = [stat.cv(np.diff(spiketrain))
                        for spiketrain in dithered_spiketrains]
        results[surr_method] = cvs_dithered
    np.save(f'{DATA_PATH}/cv_change.npy', results)


if __name__ == '__main__':
    print('firing rate change')
    firing_rate_change()
    print('cv_change')
    cv_change()
    print('clipped firing rate and eff_moved')
    clipped_firing_rate_and_eff_moved()
    print('statistical analysis of single rate')
    statistical_analysis_of_single_rate(rate=60 * pq.Hz, num_spikes=500000)
