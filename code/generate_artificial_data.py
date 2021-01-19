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


def estimate_rate_refrperiod_cv(spiketrain,
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
    cv: float
        coefficient of variation of the spiketrain
    """

    # create list of trials and deconcatenate data
    trial_list = create_st_list(spiketrain=spiketrain, sep=sep)

    # using Shinomoto rate estimation
    if all(len(trial) > 10 for trial in trial_list):
        rates = [stat.instantaneous_rate(
            spiketrain=trial,
            sampling_period=sampling_period) for trial in trial_list]
    else:
        rates = [stat.instantaneous_rate(
            spiketrain=trial,
            sampling_period=sampling_period,
            kernel=stat.kernels.GaussianKernel(
                sigma=sigma)
        ) for trial in trial_list]

    # reconcatenating rates
    rate = np.zeros(
        shape=len(trial_list) * int(
            (trial_length.simplified.magnitude +
             sep.simplified.magnitude)/sampling_period.simplified.magnitude))
    for trial_id, rate_trial in enumerate(rates):
        start_id = trial_id * int(
            ((trial_length + sep)/sampling_period).simplified.magnitude)
        stop_id = start_id + int(
            (trial_length/sampling_period).simplified.magnitude)
        rate[start_id: stop_id] = rate_trial.flatten().magnitude

    rate = neo.AnalogSignal(signal=rate,
                            units=rates[0].units,
                            sampling_period=sampling_period)

    # calculating refractory period
    refractory_period = max_refractory
    if len(spiketrain) > 1:
        isi = np.diff(spiketrain.simplified.magnitude)
        refractory_period = np.min(
            isi,
            initial=max_refractory.simplified.magnitude) * pq.s

    if len(spiketrain) > 5:
        cv = stat.cv(spiketrain)
    else:
        cv = 1

    return rate, refractory_period, cv


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
                if len(st) > 1]
        denominator = np.sum([len(trial_isi) - 1 if len(trial_isi) > 0 else 0
                    for trial_isi in isis])
        if denominator == 0:
            cv2 = 1.
        else:
            cv2 = np.sum(
                [2 * np.sum(
                    np.abs(trial_isi[:-1] - trial_isi[1:]) /
                    (trial_isi[:-1] + trial_isi[1:]))
                 for trial_isi in isis]
            ) / np.sum([len(trial_isi) - 1 if len(trial_isi) > 0 else 0
                        for trial_isi in isis])
        return cv2
    return 1.


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
    binsize: pq.Quantity
        binsize of the spade analysis
    winlen: pq.Quantity


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
    cvs = []
    cv2s = []
    for index, spiketrain in enumerate(data):
        # estimate statistics
        rate, refractory_period, cv = \
            estimate_rate_refrperiod_cv(spiketrain=spiketrain,
                                        max_refractory=max_refractory,
                                        sep=sep)
        cv2 = get_cv2(spiketrain, sep=sep)
        rates.append(rate)
        refractory_periods.append(refractory_period)
        cvs.append(cv)
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
            # generating gamma spike train with cv estimated by neuron
            shape_factor = get_shape_factor_from_cv2(cv2)
            if np.count_nonzero(rate) != 0:
                gamma_spiketrain = stg.inhomogeneous_gamma_process(
                    rate=rate, shape_factor=shape_factor)
            else:
                gamma_spiketrain = neo.SpikeTrain([]*pq.s,
                                                  t_start=spiketrain.t_start,
                                                  t_stop=spiketrain.t_stop)
            gamma_spiketrain.annotate(**spiketrain.annotations)
            gamma_spiketrains.append(gamma_spiketrain)
    return ppd_spiketrains, gamma_spiketrains, cvs


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
        if not os.path.exists(f'../data/experimental_data/ppd/{session}'):
            os.makedirs(f'../data/experimental_data/ppd/{session}')
        if not os.path.exists(f'../data/experimental_data/gamma/{session}'):
            os.makedirs(f'../data/experimental_data/gamma/{session}')
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
                        sep=sep,
                        firing_rate_threshold=firing_rate_threshold)
                else:
                    print('Loading already concatenated spiketrains')
                    sts = np.load(f'../data/concatenated_spiketrains/'
                                  f'{session}/'
                                  f'{epoch}_{trialtype}.npy',
                                  allow_pickle=True)

                print("Generating data")
                ppd, gamma, cvs = \
                        generate_artificial_data(data=sts,
                                                 seed=seed,
                                                 max_refractory=max_refractory,
                                                 processes=processes,
                                                 sep=sep)

                print('Finished data generation')

                print('Storing data...')
                if 'ppd' in processes:
                    np.save(f'../data/experimental_data/ppd/{session}/'
                            f'ppd_{epoch}_{trialtype}.npy', ppd)
                if 'gamma' in processes:
                    np.save(f'../data/experimental_data/gamma/{session}/'
                            f'gamma_{epoch}_{trialtype}.npy', gamma)
                    np.save(f'../data/experimental_data/gamma/{session}/'
                            f'cvs_{epoch}_{trialtype}.npy', cvs)
