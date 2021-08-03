"""
Generate the data for the figure containing the overview over statistical
features of the different surrogate methods.
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

import fig5_surrogate_statistics_config as cf

DATA_PATH = cf.DATA_PATH

SURR_METHODS = cf.SURR_METHODS
DATA_TYPES = cf.DATA_TYPES

STEP_DATA_TYPE = cf.STEP_DATA_TYPE

FIRING_RATE = cf.FIRING_RATE
RATES = cf.RATES
FIRING_RATES_STEP = cf.FIRING_RATES_STEP

DURATION_RATES_STEP = cf.DURATION_RATES_STEP
HIGH_NUMBER_SPIKES = cf.HIGH_NUMBER_SPIKES
LOW_NUMBER_SPIKES = cf.LOW_NUMBER_SPIKES
NUMBER_SPIKETRAINS = cf.NUMBER_SPIKETRAINS

CVS = cf.CVS

DITHER = cf.DITHER
SPADE_BIN_SIZE = cf.SPADE_BIN_SIZE

BIN_SIZE = cf.BIN_SIZE
NUM_BINS = cf.NUM_BINS
ISI_LIM = cf.ISI_LIM

TRIAL_LENGTH = cf.TRIAL_LENGTH
TRIAL_SEPARATION = cf.TRIAL_SEPARATION

DEAD_TIME = cf.DEAD_TIME
SHAPE_FACTOR = cf.SHAPE_FACTOR


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


def _create_non_stationary_spiketrain(data_type, rate, t_start, t_stop):
    sampling_period = 1.*pq.ms
    wavelength = 100.*pq.ms

    signal = rate.item() + rate.item() * \
        np.sin(2.*np.pi *
               np.arange(
                   0., t_stop.rescale(pq.ms).item(), sampling_period.item())
               / wavelength.item())

    rate = neo.AnalogSignal(
        signal=signal * pq.Hz, t_start=t_start, t_stop=t_stop,
        sampling_period=sampling_period)

    if data_type == 'Poisson':
        spiketrain = stg.inhomogeneous_poisson_process(
            rate=rate)
    elif data_type == 'PPD':
        spiketrain = stg.inhomogeneous_poisson_process(
            rate=rate,
            refractory_period=DEAD_TIME)
    elif data_type == 'Gamma':
        spiketrain = stg.inhomogeneous_gamma_process(
            rate=rate, shape_factor=SHAPE_FACTOR)
    else:
        raise ValueError('data_type should be one of Poisson, PPD, or Gamma')
    return spiketrain


def _get_dithered_spiketrains(spiketrains, surr_method):
    dithered_spiketrains = []
    for spiketrain in spiketrains:
        if surr_method == 'UD':
            dithered_spiketrain = \
                surr.dither_spikes(
                    spiketrain=spiketrain,
                    dither=DITHER)[0]
        elif surr_method == 'UDD':
            dithered_spiketrain = \
                surr.dither_spikes(
                    spiketrain=spiketrain,
                    dither=DITHER,
                    refractory_period=10 * pq.ms)[0]
        elif surr_method == 'JISI-D':
            dithered_spiketrain = surr.JointISI(
                spiketrain=spiketrain, dither=DITHER, method='window',
                cutoff=False, refractory_period=0., sigma=0.
            ).dithering()[0]
        elif surr_method == 'ISI-D':
            dithered_spiketrain = surr.JointISI(
                spiketrain=spiketrain, dither=DITHER, method='window',
                isi_dithering=True,
                cutoff=False, refractory_period=0., sigma=0.
            ).dithering()[0]
        elif surr_method == 'SHIFT-ST':
            dithered_spiketrain = surr.surrogates(
                method='trial_shifting',
                spiketrain=spiketrain, dt=DITHER, trial_length=TRIAL_LENGTH,
                trial_separation=TRIAL_SEPARATION,
            )[0]
        elif surr_method == 'BIN-SHUFF':
            dithered_spiketrain = surr.surrogates(
                method='bin_shuffling',
                spiketrain=spiketrain,
                dt=DITHER, bin_size=SPADE_BIN_SIZE,
            )[0]
        else:
            raise ValueError('surr_method is unknown')
        dithered_spiketrains.append(dithered_spiketrain)
    return dithered_spiketrains


def _isi(spiketrain, data_type, surr_method):
    isi = np.diff(spiketrain.rescale(pq.ms).magnitude)
    hist, bin_edges = np.histogram(
        a=isi,
        bins=np.arange(
            0., ISI_LIM * (1. / FIRING_RATE).rescale(pq.ms).magnitude,
            BIN_SIZE.rescale(pq.ms).magnitude)
    )

    # do a proper normalization in 1/s
    hist = hist * 1000. / len(isi)

    results = {'hist': hist,
               'bin_edges': bin_edges}

    np.save(f'{DATA_PATH}/isi_{data_type}_{surr_method}.npy', results)


def _displacement(spiketrain, dithered_spiketrain, data_type, surr_method):
    spiketrain_mag = spiketrain.rescale(pq.ms).magnitude
    dithered_spiketrain_mag = np.sort(
        dithered_spiketrain.rescale(pq.ms).magnitude)

    if len(spiketrain_mag) == len(dithered_spiketrain_mag):
        displacement = dithered_spiketrain_mag - spiketrain_mag
    else:
        displacement = dithered_spiketrain_mag \
                       - spiketrain_mag[:len(dithered_spiketrain_mag)]

    dither_mag = DITHER.rescale(pq.ms).item()
    hist, bin_edges = np.histogram(
        a=displacement,
        bins=np.arange(
            -dither_mag, dither_mag,
            BIN_SIZE.rescale(pq.ms).magnitude)
    )

    # do a proper normalization in 1/s
    hist = hist * 1000. / len(displacement)

    results = {'hist': hist,
               'bin_edges': bin_edges}

    np.save(f'{DATA_PATH}/displacement_{data_type}_{surr_method}.npy', results)


def _ac_cc(spiketrain, t_stop, data_type, surr_method,
           spiketrain2):
    binned_st = conv.BinnedSpikeTrain(spiketrain, bin_size=BIN_SIZE)
    ac_hist = corr.cross_correlation_histogram(
        binned_spiketrain_i=binned_st,
        binned_spiketrain_j=binned_st,
        window=[-NUM_BINS, NUM_BINS],
        border_correction=False,
        binary=False,
        kernel=None,
        method='speed'
    )[0]

    ac_hist_times = ac_hist.times.rescale(pq.ms).magnitude\
        + ac_hist.sampling_period.rescale(pq.ms).magnitude / 2
    ac_hist = ac_hist[:, 0].magnitude / \
        (FIRING_RATE * BIN_SIZE * t_stop).simplified.magnitude

    results = {'hist_times': ac_hist_times,
               'hist': ac_hist}

    np.save(f'{DATA_PATH}/ac_{data_type}_{surr_method}.npy',
            results)

    binned_st2 = conv.BinnedSpikeTrain(spiketrain2, bin_size=BIN_SIZE)

    cc_cross = corr.cross_correlation_histogram(
        binned_spiketrain_i=binned_st,
        binned_spiketrain_j=binned_st2,
        window=[-NUM_BINS, NUM_BINS],
        border_correction=False,
        binary=False,
        kernel=None,
        method='speed'
    )[0]

    cc_hist_times = cc_cross.times.rescale(pq.ms).magnitude \
        + cc_cross.sampling_period.rescale(pq.ms).magnitude / 2

    cc_hist = cc_cross[:, 0].magnitude / \
        (FIRING_RATE * BIN_SIZE * t_stop).simplified.magnitude

    results = {'hist_times': cc_hist_times,
               'hist': cc_hist}

    np.save(f'{DATA_PATH}/cc_{data_type}_{surr_method}.npy', results)


def _rate_eff_moved_indep(
        spiketrain, spiketrain2, clipped_rates, data_type, t_stop, eff_moved):
    clipped_rates[data_type] = \
        {'rate': (len(spiketrain) / t_stop).rescale(pq.Hz)}

    binned_st_bool = conv.BinnedSpikeTrain(
        spiketrain, bin_size=SPADE_BIN_SIZE).to_bool_array()
    binned_st = conv.BinnedSpikeTrain(
        spiketrain, bin_size=SPADE_BIN_SIZE).to_array()
    binned_st_2 = conv.BinnedSpikeTrain(
        spiketrain2, bin_size=SPADE_BIN_SIZE).to_array()

    clipped_rates[data_type]['binned'] = \
        np.sum(binned_st_bool) / len(spiketrain)

    eff_moved[data_type] = dict()
    eff_moved[data_type]['indep.'] = \
        np.sum(np.abs(binned_st - binned_st_2)) / 2 \
        / np.sqrt(len(spiketrain) * len(spiketrain2))


def _clipped_rates_and_eff_moved(
        spiketrain, dithered_spiketrain,
        clipped_rates, data_type, surr_method, eff_moved):

    binned_st = conv.BinnedSpikeTrain(
        spiketrain, bin_size=SPADE_BIN_SIZE).to_array()

    binned_dithered_st = conv.BinnedSpikeTrain(
        dithered_spiketrain, bin_size=SPADE_BIN_SIZE).to_bool_array()

    clipped_rates[data_type][surr_method] = \
        np.sum(binned_dithered_st) / len(spiketrain)

    eff_moved[data_type][surr_method] =\
        np.sum(np.abs(binned_st - binned_dithered_st)) / 2 \
        / np.sqrt(len(spiketrain) * len(dithered_spiketrain))


def _single_clipped_rate_and_eff_moved(rate):
    clipped_rates = {}
    eff_moved = {}

    t_start = 0. * pq.s
    t_stop = (LOW_NUMBER_SPIKES / rate).rescale(pq.s)

    for data_type in DATA_TYPES:
        np.random.seed(1)
        random.seed(1)

        spiketrain = _create_spiketrain(data_type, rate, t_start, t_stop)

        spiketrain2 = _create_spiketrain(data_type, rate, t_start, t_stop)

        _rate_eff_moved_indep(
            spiketrain, spiketrain2, clipped_rates, data_type,  t_stop,
            eff_moved)

        for surr_method in SURR_METHODS:
            np.random.seed(0)
            random.seed(0)

            dithered_spiketrain = _get_dithered_spiketrains(
                (spiketrain, ), surr_method)[0]

            _clipped_rates_and_eff_moved(
                spiketrain, dithered_spiketrain,
                clipped_rates, data_type, surr_method, eff_moved)
    return clipped_rates, eff_moved


def statistical_analysis_of_single_rate():
    """
    Calculate the ISI-distribution, autocorrelation, cross-correlation for
    original data and its surrogates, and save these measures.

    Returns
    -------
    None
    """

    t_start = 0. * pq.s
    t_stop = (HIGH_NUMBER_SPIKES / FIRING_RATE).rescale(pq.s)

    for data_type in DATA_TYPES:
        np.random.seed(1)
        random.seed(1)

        time_start = time.time()
        spiketrain = _create_spiketrain(
            data_type, FIRING_RATE, t_start, t_stop)
        spiketrain2 = _create_spiketrain(
            data_type, FIRING_RATE, t_start, t_stop)

        _isi(spiketrain, data_type, surr_method='original')

        _ac_cc(spiketrain, t_stop, data_type, surr_method='original',
               spiketrain2=spiketrain)

        time_stop = time.time()
        print('orig.', data_type, time_stop - time_start)

        for surr_method in SURR_METHODS:
            np.random.seed(0)
            random.seed(0)

            time_start = time.time()
            dithered_spiketrain, dithered_spiketrain2 = \
                _get_dithered_spiketrains(
                    (spiketrain, spiketrain2), surr_method)

            _isi(dithered_spiketrain, data_type, surr_method)

            _ac_cc(dithered_spiketrain, t_stop, data_type, surr_method,
                   spiketrain2=spiketrain)

            _displacement(spiketrain, dithered_spiketrain,
                          data_type, surr_method)

            time_stop = time.time()
            print(surr_method, data_type, time_stop-time_start)


def _convert_nested_defaultdict(default_dict_object):
    new_dict_object = {}
    for key in default_dict_object.keys():
        new_dict_object[key] = {}
        for key2 in default_dict_object[key].keys():
            new_dict_object[key][key2] = default_dict_object[key][key2]
    return new_dict_object


def clipped_rates_and_eff_moved():
    """
    This function calculates the clipped firing rate and the ratio of moved
    spikes and saves it.

    Returns
    -------
    None
    """
    rates = defaultdict(list)
    ratio_clipped = defaultdict(list)
    ratio_clipped_surr = defaultdict(lambda: defaultdict(list))

    ratio_indep_moved = defaultdict(list)
    ratio_moved = defaultdict(lambda: defaultdict(list))

    for rate in RATES:
        clipped_rates, eff_moved = _single_clipped_rate_and_eff_moved(rate)
        for data_type in DATA_TYPES:
            rates[data_type].append(clipped_rates[data_type]['rate'])
            ratio_clipped[data_type].append(clipped_rates[data_type]['binned'])
            ratio_indep_moved[data_type].append(eff_moved[data_type]['indep.'])
            for surr_method in SURR_METHODS:
                ratio_clipped_surr[data_type][surr_method].append(
                    clipped_rates[data_type][surr_method])
                ratio_moved[data_type][surr_method].append(
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
    This function calculates the change in firing rate profile
    after applying a surrogate method. Starting point is a spike train
    that has the first 75 ms a firing rate of 10 Hz and than of
    80 Hz, chosen similarly to Louis et al. (2010). The results are then saved.

    Returns
    -------
    None
    """

    rate_1, rate_2 = FIRING_RATES_STEP
    t_start = 0. * pq.ms
    t_stop = DURATION_RATES_STEP
    units = t_stop.units
    t_stop_mag = t_stop.magnitude

    dither_mag = DITHER.magnitude

    results = {}

    time_start = time.time()

    np.random.seed(1)
    spiketrains = []
    for _ in range(NUMBER_SPIKETRAINS):
        if STEP_DATA_TYPE == 'Poisson':
            spiketrain_1 = stg.homogeneous_poisson_process(
                rate=rate_1, t_start=t_start - DITHER,
                t_stop=t_stop + DITHER)
            spiketrain_2 = stg.homogeneous_poisson_process(
                rate=rate_2, t_start=t_start - DITHER,
                t_stop=t_stop + DITHER)
        elif STEP_DATA_TYPE == 'PPD':
            spiketrain_1 = stg.homogeneous_poisson_process(
                rate=rate_1, t_start=t_start-DITHER, t_stop=t_stop+DITHER,
                refractory_period=DEAD_TIME)
            spiketrain_2 = stg.homogeneous_poisson_process(
                rate=rate_2, t_start=t_start-DITHER, t_stop=t_stop+DITHER,
                refractory_period=DEAD_TIME)
        elif STEP_DATA_TYPE == 'Gamma':
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

    rate_original = stat.time_histogram(
        spiketrains, bin_size=BIN_SIZE, t_start=t_start, t_stop=t_stop,
        output='rate')
    results['original'] = rate_original
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
        elif surr_method == 'UDD':
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
                trial_separation=TRIAL_SEPARATION)[0]
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
                    dt=DITHER, bin_size=SPADE_BIN_SIZE)[0]
                dithered_spiketrains.append(dithered_spiketrain)
        else:
            raise ValueError('surr_method unknown')

        rate_dithered = stat.time_histogram(
            dithered_spiketrains, bin_size=BIN_SIZE,
            t_start=t_start, t_stop=t_stop, output='rate')
        print(surr_method, time.time()-time_start)

        results[surr_method] = rate_dithered

        np.save(f'{DATA_PATH}/rate_step.npy', results)


def cv_change():
    """
    Calculate the relationship of the CV from original data to that of
    the surrogates.

    Returns
    -------
    None
    """
    results = {}
    t_start = 0. * pq.s
    t_stop = (LOW_NUMBER_SPIKES / FIRING_RATE).rescale(pq.s)

    shape_factors = 1./CVS**2

    np.random.seed(0)
    spiketrains = [stg.homogeneous_gamma_process(
        a=shape_factor, b=FIRING_RATE*shape_factor,
        t_start=t_start, t_stop=t_stop)
        for shape_factor in shape_factors]
    cvs_real = [stat.cv(np.diff(spiketrain)) for spiketrain in spiketrains]
    results['cvs_real'] = cvs_real

    random.seed(1)
    np.random.seed(1)
    for surr_method in SURR_METHODS:
        dithered_spiketrains = \
            [_get_dithered_spiketrains((spiketrain,), surr_method)[0]
             for spiketrain in spiketrains]
        cvs_dithered = [stat.cv(np.diff(spiketrain))
                        for spiketrain in dithered_spiketrains]
        results[surr_method] = cvs_dithered
    np.save(f'{DATA_PATH}/cv_change.npy', results)


if __name__ == '__main__':
    print('clipped firing rate and eff_moved')
    # clipped_rates_and_eff_moved()
    print('statistical analysis of single rate')
    statistical_analysis_of_single_rate()
    print('firing rate change')
    # firing_rate_change()
    print('cv_change')
    # cv_change()
