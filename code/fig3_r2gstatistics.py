import numpy as np
import quantities as pq
from generate_artificial_data import estimate_rate_deadtime, \
    get_shape_factor_from_cv2, create_st_list
import elephant.statistics as stat
import elephant.spike_train_surrogates as surrogates
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


def plot_isi_surr(st, ax, dither, num_surr=500, show_ylabel=True,
                  show_xlabel=True, fontsize=14, legend=False):
    """
    Function calculating the ISI distributions of two example
    neurons and the ISI distribution of UD surrogates generated from the
    original neurons. Neurons represented are, with (channel-id,unit-id)
    notation, (7,1) and (16,1) for monkey N, (5,1) and (6.1) for monkey L.
    or each monkey, on the left we show the ISI distributions of two example
    neurons (blue line) and the ISI distribution of UD surrogates generated
    from the original neurons (gray line: average across 500 surrogates;
    gray band represents the first standard deviation) on a 1ms resolution.

    Parameters
    ----------
    sts: list
        list of spiketrains
    ax: plt.axes
        ax where to plot
    dither: pq.quantities
        dithering parameter of the surrogate generation
    sep: pq.quantities
        separation time between trials
    num_surr: int
        number of surrogates to generate to show the ISI distribution
    """
    # calculate isi of original spike train
    isi = stat.isi(st)
    # generate surrogates
    surr_list = surrogates.dither_spikes(st, n=num_surr,
                                         dither=dither*pq.s)
    # calculate isi of surrogates
    isis = [stat.isi(dithered_st) for dithered_st in surr_list]
    bins = np.arange(0,0.05, 0.001)
    # generate histogram of original st
    isi_distr, bin_edges = np.histogram(isi, bins=bins,
                    density=True)
    # generate histogram of surrogates
    hist_list = [
                np.histogram(
                    isi_item,
                    bins=bins,
                    density=True)[0] for isi_item in isis]
    # calculate mean and std of surrogates
    hist_mean = np.mean(hist_list, axis=0)
    hist_std = np.std(hist_list, axis=0)
    # bin coordinates
    bin_coordinates = bin_edges[:-1] + bin_edges[0] / 2
    # plot the ISI distribution of the original spike train
    ax.plot(bin_coordinates, isi_distr, label='Original')
    # plot the mean ISI distribution of the UD surrogates
    ax.fill_between(bin_coordinates, hist_mean - hist_std,
                    hist_mean + hist_std, color='lightgrey')
    ax.plot(bin_coordinates, hist_mean, label='UD surrogates', color='grey')
    if legend:
        ax.legend(fontsize=fontsize-2)
    # red bar for indication of binsize
    ax.axvline(x=bin_coordinates[5], color='navy', linestyle='--')
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    if show_xlabel:
        ax.set_xlabel('ISI (s)', fontsize=fontsize)
    if show_ylabel:
        ax.set_ylabel('Count', fontsize=fontsize)


def create_sts_list(sts, sep, epoch_length):
    """
    The function generates a list of spiketrains from the concatenated data,
    where each spiketrain is a list of trials.

    Parameters
    ----------
    sts : list of neo.SpikeTrain
        spiketrains which are concatenated over trials for a
        certain epoch.
    sep: pq.Quantity
        buffer in between concatenated trials
    epoch_length: pq.Quantity
        length of each trial

    Returns
    -------
    sts_list : list of neo.SpikeTrain
        List of spiketrains, where each spiketrain corresponds to a list of
        trials of a certain epoch.
    """
    sts_list = []

    for st in sts:
        single_concatenated_st = create_st_list(spiketrain=st,
                                                sep=sep,
                                                epoch_length=epoch_length)
        sts_list.append(single_concatenated_st)
    return sts_list


def get_cv2(isis):
    """
    Function calculating the CV2 given a list of ISIs extracted from one
    spiketrain. Function from Van Vreiswijk 2010.

    Parameters
    ----------
    isis: list
        list of ISIs
    """
    cv2s = [stat.cv2(isi, with_nan=True) for isi in isis]
    return np.nanmean(cv2s)


def plot_cv2(sts, ax, epoch_length, sep, show_xlabel=True,
             show_ylabel=True, fontsize=14):
    """
    Function producing the distribution of CV2 of all neurons in the
    considered dataset.

    Parameters
    ----------
    sts: list
        list of spiketrains
    ax: plt.axes
        ax where to plot
    epoch_length: pq.quantities
        length of each trial
    sep: pq.quantities
        separation time between trials
    """
    sts_list = create_sts_list(sts, epoch_length=epoch_length,
                               sep=sep)
    # loop over the neurons
    cv2_list = []
    for neuron, conc_st in enumerate(sts_list):
        isis = [np.diff(st.magnitude)
                for st in conc_st
                if len(st) > 1]
        cv2 = get_cv2(isis)
        cv2_list.append(cv2)
    cv2_array = np.array(cv2_list)[~np.isnan(np.array(cv2_list))]
    mean_cv2 = np.mean(np.array(cv2_array))
    shape = get_shape_factor_from_cv2(mean_cv2)
    bin_width = 0.05
    bins = np.arange(0,2, bin_width)
    ax.hist(cv2_list, bins, alpha=1)
    ax.set_xticks(np.arange(0, 1.9, 0.5))
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    if show_xlabel:
        ax.set_xlabel('CV2', fontsize=fontsize)
    if show_ylabel:
        ax.set_ylabel('Count', fontsize=fontsize)


def plot_dt(sts, ax, sorting_dead_time, sep, max_refractory=4 * pq.ms,
            show_xlabel=True, show_ylabel=True, fontsize=14):
    """
    Function producing the distribution of dead times (calculated as minimal
    ISI) of all neurons in the considered dataset.

    Parameters
    ----------
    sts: list
        list of spiketrains
    ax: plt.axes
        ax where to plot
    sep: pq.quantities
        separation time between trials
    sorting_deadtime: dict
        dictionary of dead times for all neurons fixed during spike sorting
    max_refractory: pq.quantities
        maximal refractory period as a top boundary
    """
    rp_list = []
    # loop over the neurons
    for st in sts:
        rp = estimate_rate_deadtime(st,
                                    max_refractory=max_refractory,
                                    sampling_period=1*pq.ms,
                                    sep=sep)[1]
        rp_list.append(rp * 1000)
    rp_array = np.array(rp_list)[~np.isnan(np.array(rp_list))]
    bin_width = 0.1
    sorting_dead_time = sorting_dead_time.rescale(pq.ms).magnitude
    bins = np.arange(0,4, bin_width)
    ax.set_xticks(np.arange(0, 4, 1))
    ax.hist(rp_array, bins, alpha=1)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.axvline(x=sorting_dead_time, color='grey', linestyle='--')
    if show_xlabel:
        ax.set_xlabel(r'$\tau_r$ (ms)', fontsize=fontsize)
    if show_ylabel:
        ax.set_ylabel('Count', fontsize=fontsize)


def r2g_statistics(sessions, epoch, trialtype, data_path,
                   sorting_deadtime, sep, dither, epoch_length):
    """
    Function reproducing Figure 3 of the paper. It represents the neuronal
    statistics of experimental data (ISI distribution, CV2, dead time d).
    At top: Statistics shown for Monkey N (session i140703-001, movement PGHF).
    Bottom: Statistics shown for Monkey L (session l101210-001, movement PGHF).
    For each monkey, on the left the ISI distributions of two example
    neurons and the ISI distribution of UD surrogates generated from the
    original neurons. On the right, the distribution of the average CV2 for
    all neurons and their respective dead time distribution (d).

    Parameters
    ----------
    sessions: list of strings
        sessions plotted
    epoch: str
        epoch of the trial taken into consideration
    trialtype: str
        trialtype taken into consideration
    dither: pq.quantities
        dithering parameter of the surrogate generation
    data_path: str
        data folder
    epoch_length: pq.quantities
        length of each trial
    sep: pq.quantities
        separation time between trials
    sorting_deadtime: dict
        dictionary of dead times for all neurons fixed during spike sorting

    """
    # gridspec inside gridspec
    fig = plt.figure(figsize=(5.2, 5))

    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.6)

    # figure Nikos
    # Load R2G concatenated data
    sts_N = np.load(data_path + f'{sessions[0]}/'
                    f'{epoch}_{trialtype}.npy', allow_pickle=True)

    gs0 = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=2,
                                           subplot_spec=gs[0], wspace=0.3)

    # figures ISI distribution of 2 example neurons

    gs00 = gridspec.GridSpecFromSubplotSpec(nrows=2, ncols=1,
                                            subplot_spec=gs0[0], hspace=0)
    ax01 = fig.add_subplot(gs00[0])
    plt.figtext(x=0.16, y=0.9,
                s='Monkey N - Session i140703-001 - movement PGHF',
                fontsize=10)
    plot_isi_surr(sts_N[10], ax01, dither=dither,
                  show_xlabel=False, fontsize=10, legend=True)

    # hide ticklabels of first ISI figure
    plt.setp(ax01.get_xticklabels(), visible=False)

    ax02 = fig.add_subplot(gs00[1], sharex=ax01)
    plot_isi_surr(sts_N[27], ax02, dither=dither,
                  fontsize=10)

    gs01 = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=2,
                                            subplot_spec=gs0[1], wspace=0)

    # figure CV2 histogram
    ax03 = fig.add_subplot(gs01[0])
    plot_cv2(sts_N, ax03, epoch_length=epoch_length, sep=sep, fontsize=10)

    # figure Dead time histogram
    ax04 = fig.add_subplot(gs01[1], sharey=ax03)
    plot_dt(sts_N, ax04, sorting_dead_time=sorting_deadtime['N'],
            sep=sep, show_ylabel=False,
            fontsize=10)
    plt.setp(ax04.get_yticklabels(), visible=False)

    # figure Lilou
    # Load R2G concatenated data
    sts_L = np.load(data_path + f'{sessions[1]}/'
                    f'{epoch}_{trialtype}.npy', allow_pickle=True)

    gs1 = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=2,
                                           subplot_spec=gs[1], wspace=0.3)

    # figures ISI distribution of 2 example neurons

    gs10 = gridspec.GridSpecFromSubplotSpec(nrows=2, ncols=1,
                                            subplot_spec=gs1[0], hspace=0)
    ax11 = fig.add_subplot(gs10[0])
    plt.figtext(x=0.16, y=0.45,
                s='Monkey L - Session l101210-001 - movement PGHF',
                fontsize=10)
    plot_isi_surr(sts_L[7], ax11, dither=dither,
                  show_xlabel=False, fontsize=10)

    # hide ticklabels of first ISI figure
    plt.setp(ax11.get_xticklabels(), visible=False)

    ax12 = fig.add_subplot(gs10[1], sharex=ax11)
    plot_isi_surr(sts_L[8], ax12, dither=dither,
                  fontsize=10)

    gs11 = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=2,
                                            subplot_spec=gs1[1], wspace=0)

    # figure CV2 histogram
    ax13 = fig.add_subplot(gs11[0])
    plot_cv2(sts_L, ax13, epoch_length=epoch_length, sep=sep, fontsize=10)

    # figure Dead time histogram
    ax14 = fig.add_subplot(gs11[1], sharey=ax13)
    plot_dt(sts_L, ax14, sorting_dead_time=sorting_deadtime['N'],
            sep=sep, show_ylabel=False,
            fontsize=10)
    plt.setp(ax14.get_yticklabels(), visible=False)
    if not os.path.exists('../plots'):
        os.mkdir('../plots')
    plt.savefig('../plots/fig3_r2g_statistics.eps')


if __name__ == '__main__':
    import yaml
    from yaml import Loader
    with open("configfile.yaml", 'r') as stream:
        config = yaml.load(stream, Loader=Loader)
    # The sessions to analyze
    sessions = config['sessions']
    # Magnitude of the binsize used
    binsize = config['binsize'] * pq.s
    # The winlen parameter for the SPADE analysis
    winlen = config['winlen']
    # Data being generated
    processes = config['processes']
    # SNR threshold
    SNR_thresh = config['SNR_thresh']
    # size for removal of synchrofacts from data
    synchsize = config['synchsize']
    # Firing rate threshold to exclude neurons
    firing_rate_threshold = config['firing_rate_threshold']
    # Dithering to use to generate surrogates in seconds
    dither = config['dither']

    epoch = 'movement'
    trialtype = 'PGLF'
    sep = 2 * winlen * binsize
    max_refractory = 4 * pq.ms
    sorting_deadtime = {'N': 1.2*pq.ms, 'L': 1.6*pq.ms}
    data_path = '../data/concatenated_spiketrains/'
    epoch_length = 0.5*pq.s

    r2g_statistics(sessions=sessions, epoch=epoch, trialtype=trialtype,
                   data_path=data_path, sorting_deadtime=sorting_deadtime,
                   sep=sep, dither=dither, epoch_length=epoch_length)
