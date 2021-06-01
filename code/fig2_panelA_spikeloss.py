import itertools
import os
import numpy as np
import matplotlib.pyplot as plt
import quantities as pq
from elephant import conversion
from elephant import spike_train_surrogates
import matplotlib.gridspec as gridspec


def _total_count(binned_st):
    """
    Function calculates number of spikes if the spike train.

    Parameters
    ----------
    binned_st: elephant.conversion.BinnedSpikeTrain
        Binned spike train of input

    Returns
        number of spikes of the binned spike train
    -------

    """
    binned_st = binned_st.to_array()[0, :]
    binned_st = np.array(binned_st > 0, dtype=int)
    return sum(binned_st)


def _mean_std_spike_counts_after_binning(sts, mean_list, std_list):
    """
    Function calculating a list of mean and standard deviations of the
    spike counts of the input spike trains.

    Parameters
    ----------
    sts: list
        list of spiketrains in input
    mean_list: list
        list of mean spike counts
    std_list: list
        list of std of spike counts

    Returns
    -------
    mean_list, std_list calculated inplace

    """
    spike_counts = []
    for st in sts:
        binned_st = conversion.BinnedSpikeTrain(st, binsize=binsize)
        spike_counts.append(_total_count(binned_st))
    mean_list.append(np.mean(spike_counts))
    std_list.append(np.std(spike_counts))
    return


def _mean_std_residuals(binned_single_spike_count,
                        surrogates,
                        mean_residuals,
                        std_residuals):
    """
    Function calculating a list of mean and standard deviations of the
    residuals between the one original spike train and its uniformly dithered
    surrogates.

    Parameters
    ----------
    binned_single_spike_count: int
        spike count of the considered neuron
    surrogates: list
        list of uniformly dithered surrogates of the original neuron
    mean_residuals: list
        list of mean residuals of spike counts between neurons and their UD
        surrogates
    std_residuals: list
        list of std residuals of spike counts between neurons and their UD
        surrogates

    Returns
    -------
    mean_residuals, std_residuals calculated inplace

    """
    spike_counts = []
    residuals_surrogates = []
    for st in surrogates:
        binned_st = conversion.BinnedSpikeTrain(st, binsize=binsize)
        spike_counts.append(_total_count(binned_st))
        residuals_surrogates.append(
            binned_single_spike_count - _total_count(binned_st))
    mean_residuals.append(np.mean(np.array(residuals_surrogates)))
    std_residuals.append(np.std(np.array(residuals_surrogates)))
    return


def calculate_residuals(sts, dither, binsize,
                        n_surr, epoch_length,
                        winlen):
    """
    Function calculating the residuals, their means and standard deviations
    between all spike counts of all neurons and the uniform dithering
    surrogates generated.

    Parameters
    ----------
    sts: list of neo.SpikeTrains
        spiketrains of the data set
    dither: pq.quantities
        dithering parameter of the surrogate generation
    binsize: pq.quantities
        binsize of the analysis
    n_surr: int
        number of surrogates generated
    epoch_length: pq.quantities
        length of each trial
    winlen: int
        window length of the spade analysis

    Returns
    -------
    firing_rates: list
        list of average firing rates of all neurons
    mean_residuals: list
        list of mean residuals of spike counts between neurons and their UD
        surrogates
    std_residuals: list
        list of std residuals of spike counts between neurons and their UD
        surrogates

    """

    # define separation interval between successive trials
    sep = 2 * binsize * winlen

    original_spike_counts = []
    binned_spike_counts = []
    mean_residuals = []
    std_residuals = []

    # calculate number of trials
    number_of_trials = int(sts[0].t_stop / (epoch_length + sep))

    # loop over spike trains
    for i, st in enumerate(sts):
        # bin original spike train
        binned_st = conversion.BinnedSpikeTrain(st, binsize=binsize)
        original_spike_counts.append(len(st))
        # calculate number of spikes in binned spike train
        binned_single_spike_count = _total_count(binned_st)
        binned_spike_counts.append(binned_single_spike_count)
        # dithering spikes
        surrogate_sts = spike_train_surrogates.dither_spikes(
            st, dither, n=n_surr, edges=True)
        # calculating mean and std and residuals of original to binned data
        _mean_std_residuals(
            binned_single_spike_count,
            surrogate_sts,
            mean_residuals,
            std_residuals)

    firing_rates = original_spike_counts / (number_of_trials * epoch_length)

    return firing_rates, mean_residuals, std_residuals


def plot_loss_top_panel(ax, sts, dither, binsize,
                        n_surr, epoch_length,
                        winlen, fontsize):
    """
    Function producing the top panel of figure 2, representing the spike
    count reduction resulting by clipping and UD surrogate generation.
    with crosses of different colors, we represent the reduction in the spike
    count in function of the average firing rate.
    Each cross represent one neuron. In blue, we indicate the spike count
    reduction caused by clipping the original spike train.
    In orange, we indicate the spike count reduction followed by generating
    surrogates by UD and then clipping. The spike count reduction is
    expressed in percentage with respect to the original continuous-time
    spike train. Bars indicate the standard deviation of the spike count
    reduction calculated across n_surr surrogates.

    Parameters
    ----------
    ax: plt.axes
        ax of the panel
    sts: list of neo.SpikeTrains
        spiketrains of the data set
    dither: pq.quantities
        dithering parameter of the surrogate generation
    binsize: pq.quantities
        binsize of the analysis
    n_surr: int
        number of surrogates generated
    epoch_length: pq.quantities
        length of each trial
    winlen: int
        window length of the spade analysis
    fontsize: int
        fontsize for plotting

    """

    # define separation interval between trials
    sep = 2 * binsize * winlen

    original_spike_counts = []
    binned_spike_counts = []
    mean_spike_counts_dither = []
    std_spike_counts_dither = []

    number_of_trials = int(sts[0].t_stop / (epoch_length + sep))
    for i, st in enumerate(sts):
        binned_st = conversion.BinnedSpikeTrain(st, binsize=binsize)
        original_spike_counts.append(len(st))
        binned_spike_counts.append(_total_count(binned_st))

        surrogate_sts = spike_train_surrogates.dither_spikes(
            st, dither, n=n_surr, edges=True)
        _mean_std_spike_counts_after_binning(
            surrogate_sts, mean_spike_counts_dither,
            std_spike_counts_dither)

    original_spike_counts = np.array(original_spike_counts)
    binned_spike_counts = np.array(binned_spike_counts)

    mean_spike_counts_dither = np.array(mean_spike_counts_dither)
    std_spike_counts_dither = np.array(std_spike_counts_dither)

    firing_rates = original_spike_counts / (number_of_trials * epoch_length)
    binned_loss = 1. - binned_spike_counts / original_spike_counts
    mean_loss_dither = 1. - mean_spike_counts_dither / original_spike_counts
    std_loss_dither = std_spike_counts_dither / original_spike_counts

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']
    ax.scatter(firing_rates, binned_loss, label='Original + clipping',
               color=colors[0], marker='x')

    ax.errorbar(firing_rates, mean_loss_dither, yerr=std_loss_dither,
                fmt='o', label='UD Surrogate + clipping', color=colors[1],
                marker='x')
    ax.set_ylabel('Spike count decrease %', fontsize=fontsize)
    # ax.set_xlim(left=2. / epoch_length.magnitude)
    ax.set_xlim(left=2. / epoch_length.magnitude, right=65)
    ax.set_ylim(bottom=-0.01, top=0.17)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(fontsize=fontsize - 2)


def plot_residuals(ax, sts, dither, binsize, fontsize, epoch_length,
                   winlen, n_surr):
    """
    Function producing the lower panel of Figure 2. Residuals computed as
    the difference between the original clipped spike trains and the UD
    binned and clipped surrogates, i.e., between the blue and the orange
    crosses.

    Parameters
    ----------
    ax: plt.axes
        ax of the panel
    sts: list of neo.SpikeTrains
        spiketrains of the data set
    dither: pq.quantities
        dithering parameter of the surrogate generation
    binsize: pq.quantities
        binsize of the analysis
    n_surr: int
        number of surrogates generated
    epoch_length: pq.quantities
        length of each trial
    winlen: int
        window length of the spade analysis

    """

    firing_rates, mean_residuals, std_residuals = \
        calculate_residuals(sts,
                            dither=dither,
                            binsize=binsize,
                            n_surr=n_surr,
                            epoch_length=epoch_length,
                            winlen=winlen)
    ax.errorbar(firing_rates, mean_residuals, yerr=std_residuals,
                fmt='o', label='UD Surrogate + clipping', color='grey',
                marker='x')
    ax.set_xlabel('Average Firing rate (Hz)', fontsize=fontsize)
    ax.set_ylabel('Residuals', fontsize=fontsize)
    ax.set_xlim(left=0, right=65)
    ax.set_ylim(bottom=0, top=100)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_xticks(np.arange(0, 65, 20))


def plot_fig2_spikeloss(
        folder,
        sessions,
        epoch,
        trialtype,
        dither,
        binsize,
        n_surr,
        winlen,
        epoch_length,
        fontsize,
        data_type='original',
        file_type='eps'
):
    """
    Function to reproduce fig 2 of the paper.
    Spike count reduction resulting by clipping and UD surrogate generation.
    Results from the analysis of two experimental data sets, in the movement
    epoch of trial type precision grip-high force (PGHF) of monkeys N (left)
    and L (right). Top panel: we represent the reduction in the spike count
    in function of the average firing rate of all neurons.The spike count
    reduction is expressed in percentage with respect to the original
    continuous-time spike train. Bars indicate the standard deviation of the
    spike count reduction calculated across 100 surrogates. Bottom panel:
    residuals computed as the difference between the original clipped spike
    trains and the UD binned and clipped surrogates.

    Parameters
    ----------
    folder: folder
        data folder
    sessions: list of strings
        sessions plotted
    epoch: str
        epoch of the trial taken into consideration
    trialtype: str
        trialtype taken into consideration
    dither: pq.quantities
        dithering parameter of the surrogate generation
    binsize: pq.quantities
        binsize of the analysis
    n_surr: int
        number of surrogates generated
    epoch_length: pq.quantities
        length of each trial
    winlen: int
        window length of the spade analysis
    data_type: {'original', 'ppd', 'gamma'}, optional
        Default: 'original'
    file_type: {'eps', 'png'}, optional
        Default: 'eps'

    """
    np.random.seed(0)
    # gridspec inside gridspec
    fig = plt.figure(figsize=(7.5, 5))

    gs = gridspec.GridSpec(1, 2, figure=fig, hspace=0.4, wspace=0.35, top=0.87,
                           left=0.11, right=0.97)

    # Nikos
    # loading
    extra_part = f'{data_type}_' if data_type != 'original' else ''
    file_nikos = f'{folder}{sessions[0]}/{extra_part}{epoch}_{trialtype}.npy'
    sts_N = np.load(
        file_nikos,
        allow_pickle=True)

    # plotting
    gs0 = gridspec.GridSpecFromSubplotSpec(nrows=2, ncols=1,
                                           subplot_spec=gs[0], hspace=0,
                                           height_ratios=[2, 1])
    ax01 = fig.add_subplot(gs0[0])
    plot_loss_top_panel(ax=ax01,
                        sts=sts_N,
                        binsize=binsize,
                        dither=dither,
                        n_surr=n_surr,
                        epoch_length=epoch_length,
                        winlen=winlen,
                        fontsize=fontsize)
    # hide ticklabels of first ISI figure
    plt.setp(ax01.get_xticklabels(), visible=False)
    ax02 = fig.add_subplot(gs0[1], sharex=ax01)
    plot_residuals(ax=ax02,
                   sts=sts_N,
                   binsize=binsize,
                   dither=dither,
                   epoch_length=epoch_length,
                   winlen=winlen,
                   n_surr=n_surr,

                   fontsize=fontsize)
    if data_type == 'original':
        title = f'Monkey N\n' \
                f'Session {sessions[0]}, {epoch} {trialtype}'
    else:
        title = f'Monkey N - {data_type}\n' \
               f'Session {sessions[0]}, {epoch} {trialtype}'
    plt.figtext(
        x=0.12, y=0.9, s=title, fontsize=10, multialignment='center')

    # Lilou
    sts_L = np.load(
        f'{folder}{sessions[1]}/{extra_part}{epoch}_{trialtype}.npy',
        allow_pickle=True)

    gs1 = gridspec.GridSpecFromSubplotSpec(nrows=2, ncols=1,
                                           subplot_spec=gs[1], hspace=0,
                                           height_ratios=[2, 1])
    ax11 = fig.add_subplot(gs1[0])
    plot_loss_top_panel(ax=ax11,
                        sts=sts_L,
                        binsize=binsize,
                        dither=dither,
                        fontsize=fontsize,
                        n_surr=n_surr,
                        epoch_length=epoch_length,
                        winlen=winlen)
    # hide ticklabels of first ISI figure
    plt.setp(ax11.get_xticklabels(), visible=False)
    ax12 = fig.add_subplot(gs1[1], sharex=ax11)
    plot_residuals(ax=ax12,
                   sts=sts_L,
                   binsize=binsize,
                   dither=dither,
                   epoch_length=epoch_length,
                   winlen=winlen,
                   n_surr=n_surr,
                   fontsize=fontsize)
    if data_type == 'original':
        title = f'Monkey L\n' \
                f'Session {sessions[1]}, {epoch} {trialtype}'
    else:
        title = f'Monkey L - {data_type}\n' \
               f'Session {sessions[1]}, {epoch} {trialtype}'
    plt.figtext(
        x=0.61, y=0.9, s=title, fontsize=10, multialignment='center')

    plt.rcParams.update({'font.size': 10})
    if not os.path.exists('../plots/fig2_spikeloss'):
        os.mkdir('../plots/fig2_spikeloss')
    plt.savefig(
        f'../plots/fig2_spikeloss/'
        f'fig2_spikeloss_{epoch}_{trialtype}_{data_type}.{file_type}')
    plt.close(fig=fig)


if __name__ == '__main__':
    import yaml
    from yaml import Loader
    with open("configfile.yaml", 'r') as stream:
        config = yaml.load(stream, Loader=Loader)
    # Unit in which every time of the analysis is expressed
    unit = config['unit']
    # Magnitude of the binsize used
    binsize = (config['binsize'] * pq.s).rescale(pq.ms)
    # The sessions to analyze
    sessions = config['sessions']
    # Dithering to use to generate surrogates in seconds
    dither = config['dither'] * pq.s
    # The winlen parameter for the SPADE analysis
    winlen = config['winlen']
    # SNR threshold
    SNR_thresh = config['SNR_thresh']
    # size for removal of synchrofacts from data
    synchsize = config['synchsize']

    data_types = ['original']
    data_types.extend(config['processes'])

    data_type_fixed = 'original'
    epoch_fixed = 'movement'
    trialtype_fixed = 'PGHF'

    n_surr = 100
    epoch_length = 0.5 * pq.s
    fontsize = 10

    # file_type = 'png'
    file_type = 'eps'  # for publication in PLOS CB

    for epoch, trialtype, data_type in itertools.product(
            config['epochs'], config['trialtypes'], data_types):
        if not (data_type == data_type_fixed
                and trialtype == trialtype_fixed
                and epoch == epoch_fixed):
            continue

        if data_type == 'original':
            sts_folder = '../data/concatenated_spiketrains/'
        else:
            sts_folder = f'../data/artificial_data/{data_type}/'

        plot_fig2_spikeloss(
            folder=sts_folder,
            sessions=sessions,
            epoch=epoch,
            trialtype=trialtype,
            dither=dither,
            binsize=binsize,
            n_surr=n_surr,
            winlen=winlen,
            epoch_length=epoch_length,
            fontsize=fontsize,
            data_type=data_type,
            file_type=file_type)
