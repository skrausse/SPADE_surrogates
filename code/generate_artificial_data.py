# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import quantities as pq
import elephant.spike_train_generation as stg
import elephant.statistics as stat
sys.path.insert(0, '../data/multielectrode_grasp/code/python-neo')
import neo
sys.path.append('./')
from scipy.special import gamma as gamma_function
from scipy.optimize import brentq
from tqdm import tqdm


def estimate_rate_deadtime(
        spiketrain,
        max_refractory,
        sep,
        sampling_period=0.1 * pq.ms,
        sigma=100 * pq.ms,
        trial_length=500 * pq.ms):
    """
    Function estimating rate, refractory period and cv, given one spiketrain
    Parameters
    ----------
    spiketrain : neo.SpikeTrain
        spiketrain from which rate, refractory period and cv are estimated
    max_refractory : pq.Quantity
        maximal refractory period allowed
    sep : pq.Quantity
        buffer in between trials in the concatenated data (typically is equal
        to 2 * binsize * winlen)
    sampling_period : pq.Quantity
        sampling period of the firing rate (optional)
    sigma : pq.Quantity
        sd of the gaussian kernel for rate estimation (optional)
    trial_length : pq.Quantity
        duration of each trial
    Returns
    -------
    rate: neo.AnalogSignal
        rate of the spiketrain
    refractory_period: pq.Quantity
        refractory period of the spiketrain (minimal isi)
    rate_list: list
        list of rate profiles for all trials
    """

    # create list of trials and deconcatenate data
    trial_list = create_st_list(spiketrain=spiketrain, sep=sep)

    # using Shinomoto rate estimation
    if all(len(trial) > 10 for trial in trial_list):
        rate_list = [stat.instantaneous_rate(
            spiketrain=trial,
            sampling_period=sampling_period,
            border_correction=True) for trial in trial_list]
    else:
        rate_list = [stat.instantaneous_rate(
            spiketrain=trial,
            sampling_period=sampling_period,
            kernel=stat.kernels.GaussianKernel(
                sigma=sigma),
            border_correction=True
        ) for trial in trial_list]

    # reconcatenating rates
    rate = np.zeros(
        shape=len(trial_list) * int(
            (trial_length.simplified.magnitude +
             sep.simplified.magnitude)/sampling_period.simplified.magnitude))
    for trial_id, rate_trial in enumerate(rate_list):
        start_id = trial_id * int(
            ((trial_length + sep)/sampling_period).simplified.magnitude)
        stop_id = start_id + int(
            (trial_length/sampling_period).simplified.magnitude)
        rate[start_id: stop_id] = rate_trial.flatten().magnitude

    rate = neo.AnalogSignal(signal=rate,
                            units=rate_list[0].units,
                            sampling_period=sampling_period)

    # calculating refractory period
    refractory_period = estimate_deadtime(
        spiketrain, max_dead_time=max_refractory)

    return rate, refractory_period, rate_list


def estimate_deadtime(spiketrain, max_dead_time):
    """
    Function to calculate the dead time of one spike train.
    
    Parameters
    ----------
    spiketrain : neo.SpikeTrain
        spiketrain from which rate, refractory period and cv are estimated
    max_dead_time : pq.Quantity
        maximal refractory period allowed

    Returns
    -------
    dead_time: pq.Quantity
        refractory period of the spiketrain (minimal isi)
    """
    if len(spiketrain) > 1:
        isi = np.diff(spiketrain.simplified.magnitude)
        dead_time = np.min(
            isi, initial=max_dead_time.simplified.magnitude) * pq.s
    else:
        dead_time = max_dead_time

    return dead_time


def create_st_list(spiketrain, sep,
                   epoch_length=0.5*pq.s):
    """
    The function generates a list of spiketrains from the concatenated data,
    where each list corresponds to a trial.

    Parameters
    ----------
    spiketrain : neo.SpikeTrain
        spiketrains which are concatenated over trials for a
        certain epoch.
    sep: pq.Quantity
        buffer in between concatenated trials
    epoch_length: pq.Quantity
        length of each trial
    Returns
    -------
    spiketrain_list : list of neo.SpikeTrain
        List of spiketrains, where each spiketrain corresponds
        to one trial of a certain epoch.
    """
    spiketrain_list = []

    t_max = spiketrain.t_stop

    trial = 0
    t_start = 0.0 * pq.s
    sep = sep.rescale(pq.s)
    epoch_length = epoch_length.rescale(pq.s)
    while t_start < t_max - sep:
        t_start = trial * (epoch_length + sep)
        t_stop = trial * (epoch_length + sep) + epoch_length

        if t_start > t_max - sep:
            break
        cutted_st = spiketrain[np.all(
            [spiketrain > t_start, spiketrain < t_stop], axis=0)] - t_start
        cutted_st.t_start = 0.0 * pq.s
        cutted_st.t_stop = epoch_length
        spiketrain_list.append(cutted_st)
        trial += 1
    return spiketrain_list


def get_cv2(spiketrain, sep):
    """
    calculates the cv2 of a spiketrain

    Parameters
    ----------
    spiketrain: neo.SpikeTrain
        single neuron concatenate spike train
    sep: pq.Quantity

    Returns
    -------
    cv2 : float
        The CV2 or 1. if not enough spikes are in the spiketrain.
    """
    if len(spiketrain) > 5:
        spiketrain_list = create_st_list(spiketrain, sep=sep)
        isis = [np.diff(st.magnitude)
                for st in spiketrain_list
                if len(st) > 2]
        cv2_list = [stat.cv2(isi, with_nan=True) for isi in isis]
        return np.nanmean(cv2_list)
    return np.nan


def get_cv2_from_shape_factor(shape_factor):
    """
    calculates cv2 from shape factor based on van Vreswijk's formula

    Parameters
    ----------
    shape_factor : float

    Returns
    -------
    cv2 : float
    """
    return gamma_function(2 * shape_factor) / \
        (shape_factor * (2 ** (shape_factor - 1)
                         * gamma_function(shape_factor))) ** 2


def get_shape_factor_from_cv2(cv2):
    """
    calculates shape factor from cv2 based on van Vreswijk's formula

    Parameters
    ----------
    cv2 : float

    Returns
    -------
    shape_factor : float
    """

    return brentq(lambda x: get_cv2_from_shape_factor(x) - cv2, 0.05, 50.)


def get_cv_operational_time(spiketrain, rate_list, sep):
    """
    calculates cv of spike train in operational time

    Parameters
    ----------

    Returns
    -------
    cv : float
    """
    # deconcatenate spike train into list of trials
    trial_list = create_st_list(spiketrain, sep=sep)
    isis_operational_time = []

    # requiring at least one spike per trial, if not returning cv=1
    if sum(len(trial)for trial in trial_list) < len(trial_list):
        return 1

    for rate, trial in zip(rate_list, trial_list):
        # check there is at least one ISI per trial
        if len(trial) > 1:
            # The time points at which the firing rates are given
            real_time = np.hstack((rate.times.simplified.magnitude,
                                   rate.t_stop.simplified.magnitude))
            # indices where between which points in real time the spikes lie
            indices = np.searchsorted(real_time, trial)

            # Operational time corresponds to the integral of the firing rate over time
            operational_time = np.cumsum(
                (rate*rate.sampling_period).simplified.magnitude)
            operational_time = np.hstack((0., operational_time))
            # In real time the spikes are first aligned to the left border of the bin.
            trial_operational_time = operational_time[indices - 1]
            # the relative position of the spikes in the operational time bins
            positions_in_bins = \
                (trial_operational_time - real_time[indices - 1]) / \
                rate.sampling_period.simplified.magnitude
            # add the positions in the bin times the sampling period in op time
            trial_operational_time += \
                (operational_time[indices] - operational_time[indices-1]) \
                * positions_in_bins
            # add isis per trial into the overall list
            isis_operational_time.append(np.diff(trial_operational_time))

    number_of_isis = np.sum([len(trial_isi) for trial_isi
                             in isis_operational_time])
    cv = 1
    if number_of_isis > 3:
        mean_isi = \
            np.sum([np.sum(trial_isi) for trial_isi in
                    isis_operational_time]) / \
            number_of_isis

        variance_isi = np.sum(
            [np.sum((trial_isi - mean_isi) ** 2) for trial_isi
             in isis_operational_time]) / \
                       number_of_isis
        cv = np.sqrt(variance_isi) / mean_isi

    return cv


def generate_artificial_data(data, seed, max_refractory, processes,
                             sep):
    """
    Generate data as Poisson with refractory period and Gamma
    Parameters
    ----------
    data: list
        list of spiketrains
    seed: int
        seed for the data generation
    max_refractory: quantity
        maximal refractory period
    processes: list
        processes to be generated
    sep: pq.Quantity
        buffering between two trials


    Returns
    -------
    ppd_spiketrains: list
        list of poisson processes (neo.SpikeTrain) with rate profile and
        refractory period estimated from data
    gamma_spiketrains: list
        list of gamma processes (neo.SpikeTrain) with rate profile and
        shape (cv) estimated from data
    cv_list : list
        list of cvs, estimated from data, one for each neuron

    """

    # Setting random seed globally
    np.random.seed(seed=seed)

    rates = []
    refractory_periods = []
    ppd_spiketrains = []
    gamma_spiketrains = []
    cv2s = []

    for index, spiketrain in tqdm(enumerate(data)):
        # estimate statistics
        rate, refractory_period, rate_list = \
            estimate_rate_deadtime(spiketrain=spiketrain,
                                   max_refractory=max_refractory,
                                   sep=sep)
        cv2 = get_cv2(spiketrain, sep=sep)
        rates.append(rate)
        refractory_periods.append(refractory_period)
        cv2s.append(cv2)

        if 'ppd' in processes:
            # generating Poisson spike trains with refractory period
            ppd_spiketrain = stg.inhomogeneous_poisson_process(
                rate=rate,
                as_array=False,
                refractory_period=refractory_period)
            ppd_spiketrain.annotate(**spiketrain.annotations)
            ppd_spiketrain = ppd_spiketrain.rescale(pq.s)
            ppd_spiketrains.append(ppd_spiketrain)

        if 'gamma' in processes:
            cv = get_cv_operational_time(spiketrain=spiketrain,
                                                      rate_list=rate_list,
                                                      sep=sep)
            shape_factor = 1/(cv)**2
            print('cv = ', cv, 'shape_factor = ', shape_factor)
            if np.count_nonzero(rate) != 0:
                gamma_spiketrain = stg.inhomogeneous_gamma_process(
                    rate=rate, shape_factor=shape_factor)
            else:
                gamma_spiketrain = neo.SpikeTrain([]*pq.s,
                                                  t_start=spiketrain.t_start,
                                                  t_stop=spiketrain.t_stop)
            gamma_spiketrain.annotate(**spiketrain.annotations)
            gamma_spiketrains.append(gamma_spiketrain)
    return ppd_spiketrains, gamma_spiketrains, cv2s


if __name__ == '__main__':
    import rgutils
    import yaml
    from yaml import Loader

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
    processes = config['processes']
    SNR_thresh = 2.5
    synchsize = 2
    sep = 2 * winlen * binsize
    max_refractory = 4 * pq.ms
    load_original_data = False

    for session in sessions:
        if not os.path.exists(f'../data/artificial_data/ppd/{session}'):
            os.makedirs(f'../data/artificial_data/ppd/{session}')
        if not os.path.exists(f'../data/artificial_data/gamma/{session}'):
            os.makedirs(f'../data/artificial_data/gamma/{session}')
        for epoch in epochs:
            for trialtype in trialtypes:
                print(f'Loading data {session} {epoch} {trialtype}')
                if load_original_data:
                    print('Loading experimental data')
                    sts = rgutils.load_epoch_concatenated_trials(
                        session,
                        epoch,
                        trialtypes=trialtype,
                        SNRthresh=SNR_thresh,
                        synchsize=synchsize,
                        sep=sep)
                else:
                    print('Loading already concatenated spiketrains')
                    sts = np.load(f'../data/concatenated_spiketrains/'
                                  f'{session}/'
                                  f'{epoch}_{trialtype}.npy',
                                  allow_pickle=True)

                print("Generating data")
                ppd, gamma, cv2s = \
                    generate_artificial_data(data=sts,
                                             seed=seed,
                                             max_refractory=max_refractory,
                                             processes=processes,
                                             sep=sep)

                print('Finished data generation')

                print('Storing data...')
                if 'ppd' in processes:
                    np.save(f'../data/artificial_data/ppd/{session}/'
                            f'ppd_{epoch}_{trialtype}.npy', ppd)
                if 'gamma' in processes:
                    np.save(f'../data/artificial_data/gamma/{session}/'
                            f'gamma_{epoch}_{trialtype}.npy', gamma)
                    np.save(f'../data/artificial_data/gamma/{session}/'
                            f'cv2s_{epoch}_{trialtype}.npy', cv2s)
