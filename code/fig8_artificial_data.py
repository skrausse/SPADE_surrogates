import numpy as np
import quantities as pq
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.pylab as pylab
from generate_artificial_data import estimate_rate_refrperiod_cv, \
    create_st_list
import elephant
import os


def build_dicts(st, gamma_st, ppd_st, max_refractory):
    """
    Function creating for each neuron dictionaries of rate, deadtime and cv
    for gamma and ppd process respectively, and as well for the original data.

    Parameters
    ----------
    st: neo.Spiketrain
        spiketrain of original experimental data
    gamma_st: neo.Spiketrain
        artificial spiketrain generated with a gamma model
    ppd_st: neo.Spiketrain
        artificial spiketrain generated with a Poisson process with dead time
        model
    max_refractory: pq.quantity
        maximal refractory period allowed in the ppd model

    Returns
    -------
    rate_dict: dict
        dictionary containing for each process (original, gamma, ppd) the rate
        estimated with a Shinomoto optimized kernel
    rp_dict: dict
        dictionary containing for each process the estimated dead time
    cv_dict: dict
        dictionary containing for each process the estimated cv
    """
    rate_dict = {}
    rp_dict = {}
    cv_dict = {}
    list_st = [st, gamma_st, ppd_st]
    processes = ['original', 'gamma', 'ppd']
    for order, process in enumerate(processes):
        rate, refractory_period, cv = \
            estimate_rate_refrperiod_cv(list_st[order],
                                        max_refractory=max_refractory,
                                        sampling_period=0.1*pq.ms)
        rate_dict[process] = rate
        rp_dict[process] = refractory_period
        cv_dict[process] = cv
    return rate_dict, rp_dict, cv_dict


def cut_firing_rate_into_trials(rate, epoch_length, sep, sampling_period):
    """
    The function generates a numpy array of rates from the concatenated rate,
    where each item corresponds to a trial.

    Parameters
    ----------
    rate : neo.SpikeTrain
        firing rate estimated for a single neuron, estimated on concatenated
        data
    sep: pq.Quantity
        buffer in between concatenated trials
    epoch_length: pq.Quantity
        length of each trial
    sampling_period: pq.Quantity
        sampling period of the rate estimation
    Returns
    -------
    trials_rate : np.array
        np.array of trial rates, where each item corresponds
        to one trial of a certain epoch.
    """
    len_rate = len(rate)
    start_index = 0
    trials_rate = []
    epoch_length = epoch_length.rescale(pq.ms)
    sep = sep.rescale(pq.ms)
    sampling_period = sampling_period.rescale(pq.ms)
    stop_index = int(start_index + (
            epoch_length/sampling_period).simplified.magnitude)
    while stop_index < len_rate:
        cut_rate = rate[start_index: stop_index]
        trials_rate.append(cut_rate)
        start_index += int((epoch_length/sampling_period).simplified.magnitude
                           + (sep/sampling_period).simplified.magnitude)
        stop_index = int(start_index + (
                epoch_length/sampling_period).simplified.magnitude)
    return np.array(trials_rate)


def plot_trial_firing_rate(ax, sts, gamma, ppd, neuron, max_refractory, sep,
                           epoch_length, sampling_period,
                           fontsize):
    """
    Plot representing the firing rate of one neuron convolved with a Shinomoto
    optimized kernel. For each neuron it is plotted the firing rate of the
    original data, the one of the corresponding data with a PPD model and with
    a gamma model. Standard deviations across trials are indicated with a
    colored band.

    Parameters
    ----------
    ax: matplotlib.pyplot.axes
        ax where to plot the figure
    sts: list
        list of neo spiketrains of the original data
    gamma: list
        list of neo spiketrains of the gamma data
    ppd: list
        list of neo spiketrains of the PPD data
    neuron: int
        index of the neuron being plotted
    max_refractory: quantity
        maximal refractory period
    sep: quantity
        separation time within trials
    epoch_length: quantity
        trial duration (typically 500pq.ms)
    sampling_period: quantity
        sampling period of the recording (30.000Hz)
    fontsize: int
        fontsize of the legend

    """
    st = sts[neuron]
    gamma_st = gamma[neuron]
    ppd_st = ppd[neuron]
    rate_dict = {}
    rp_dict = {}
    cv_dict = {}
    list_st = [st, gamma_st, ppd_st]
    processes = ['original', 'gamma', 'ppd']
    for order, process in enumerate(processes):
        rate, refractory_period, cv = \
            estimate_rate_refrperiod_cv(list_st[order],
                                        max_refractory=max_refractory,
                                        sampling_period=sampling_period,
                                        sep=sep)
        rate_dict[process] = rate
        rp_dict[process] = refractory_period
        cv_dict[process] = cv
    cut_trials_original = cut_firing_rate_into_trials(rate_dict['original'],
                                                      epoch_length=epoch_length,
                                                      sep=sep,
                                                      sampling_period=sampling_period)
    cut_trials_gamma = cut_firing_rate_into_trials(rate_dict['gamma'],
                                                   epoch_length=epoch_length,
                                                   sep=sep,
                                                   sampling_period=sampling_period)
    cut_trials_ppd = cut_firing_rate_into_trials(rate_dict['ppd'],
                                                 epoch_length=epoch_length,
                                                 sep=sep,
                                                 sampling_period=sampling_period)
    x = np.arange(0, int(epoch_length.rescale(pq.ms).magnitude/
                         sampling_period.rescale(pq.ms).magnitude))
    ax.plot(np.squeeze(np.mean(cut_trials_original, axis=0)), label='original')
    ax.fill_between(x, np.squeeze(np.mean(cut_trials_original, axis=0) - np.std(cut_trials_original, axis=0)),
                    np.squeeze(np.mean(cut_trials_original, axis=0) + np.std(cut_trials_original, axis=0)),
                    alpha=0.2)
    ax.plot(np.squeeze(np.mean(cut_trials_gamma, axis=0)), label='gamma')
    ax.fill_between(x, np.squeeze(np.mean(cut_trials_gamma, axis=0) - np.std(cut_trials_gamma, axis=0)),
                    np.squeeze(np.mean(cut_trials_gamma, axis=0) + np.std(cut_trials_gamma, axis=0)),
                    alpha=0.2)
    ax.plot(np.squeeze(np.mean(cut_trials_ppd, axis=0)), label='ppd')
    ax.fill_between(x, np.squeeze(np.mean(cut_trials_ppd, axis=0) - np.std(cut_trials_ppd, axis=0)),
                    np.squeeze(np.mean(cut_trials_ppd, axis=0) + np.std(cut_trials_ppd, axis=0)),
                    alpha=0.2)
    plt.ylim(min(np.squeeze(np.mean(cut_trials_ppd, axis=0) - np.std(cut_trials_ppd, axis=0))) - 10,
                max(np.squeeze(np.mean(cut_trials_ppd, axis=0) + np.std(cut_trials_ppd, axis=0))) + 10)
    ax.set_xlabel('time (ms)', fontsize=fontsize)
    ax.set_ylabel('firing rate (Hz)', fontsize=fontsize)
    ax.set_title('Single unit average FR', fontsize=fontsize)
    # ax.legend(fontsize=fontsize-2)
    xticks = ax.get_xticks().tolist()
    rescaled_xticks = [int(int(lab) / 10) for lab in xticks]
    ax.set_xticklabels(rescaled_xticks)
    return ax


def plot_cv(ax, sts, gamma, ppd, max_refractory, sampling_period):
    """
    Plot representing the coefficient of variation of all neurons of one
    dataset, for the original data, the gamma and the ppd data.

    Parameters
    ----------
    ax: matplotlib.pyplot.axes
        ax where to plot the figure
    sts: list
        list of neo spiketrains of the original data
    gamma: list
        list of neo spiketrains of the gamma data
    ppd: list
        list of neo spiketrains of the PPD data
    max_refractory: quantity
        maximal refractory period
    """
    processes = {'original': sts, 'ppd': ppd, 'gamma': gamma}
    # loop over the neurons
    cv_dict = {'original':[], 'ppd':[], 'gamma':[]}
    for neuron in range(len(sts)):
        for key in cv_dict.keys():
            cv = estimate_rate_refrperiod_cv(processes[key][neuron],
                                             max_refractory=max_refractory,
                                             sampling_period=sampling_period)[2]
            cv_dict[key].append(cv)
    bins = np.arange(0,1.5, 0.1)
    for key in cv_dict.keys():
        ax.hist(cv_dict[key], bins, alpha=1, label=key, histtype='step')
    ax.legend(loc='upper right')
    return ax


def plot_rp(ax, sts, gamma, ppd, max_refractory, sampling_period, sep,
            fontsize):
    """
    Plot representing the dead time of all neurons of one
    dataset, for the original data, the gamma and the ppd data.

    Parameters
    ----------
    ax: matplotlib.pyplot.axes
        ax where to plot the figure
    sts: list
        list of neo spiketrains of the original data
    gamma: list
        list of neo spiketrains of the gamma data
    ppd: list
        list of neo spiketrains of the PPD data
    max_refractory: quantity
        maximal refractory period
    sep: quantity
        separation time within trials
    epoch_length: quantity
        trial duration (typically 500pq.ms)
    sampling_period: quantity
        sampling period of the recording (30.000Hz)
    fontsize: int
        fontsize of the legend
    """
    processes = {'original': sts, 'ppd': ppd, 'gamma': gamma}
    # loop over the neurons
    rp_dict = {'original':[], 'ppd':[], 'gamma':[]}
    for neuron in range(len(sts)):
        for key in rp_dict.keys():
            rp = estimate_rate_refrperiod_cv(processes[key][neuron],
                                             max_refractory=max_refractory,
                                             sampling_period=sampling_period,
                                             sep=sep)[1]
            rp_dict[key].append(rp*1000)
    bins = np.arange(0,4, 0.1)
    for key in rp_dict.keys():
        ax.hist(rp_dict[key], bins, alpha=1, label=key, histtype='step')
    ax.legend(loc='upper right', fontsize=fontsize-4)
    ax.set_title('Dead time of all units', fontsize=fontsize)
    ax.set_ylabel('Count', fontsize=fontsize)
    ax.set_xlabel('DT (ms)', fontsize=fontsize)
    ax.set_ylim([0, max([max(np.histogram(rp_dict[key], bins)[0]) for key in rp_dict.keys()]) + 5])
    return ax


def plot_isi(ax, sts, gamma, ppd, neuron, fontsize):
    """
    Plot representing the ISI of one neurons of one
    dataset, for the original data, the gamma and the ppd data.

    Parameters
    ----------
    ax: matplotlib.pyplot.axes
        ax where to plot the figure
    sts: list
        list of neo spiketrains of the original data
    gamma: list
        list of neo spiketrains of the gamma data
    ppd: list
        list of neo spiketrains of the PPD data
    neuron: int
        index of the neuron being plotted
    fontsize: int
        fontsize of the legend
    """
    st = sts[neuron]
    gamma_st = gamma[neuron]
    ppd_st = ppd[neuron]
    isi_dict = {'original':[], 'ppd':[], 'gamma':[]}
    list_st = [st, ppd_st, gamma_st]
    for index, key in enumerate(isi_dict.keys()):
        isi = elephant.statistics.isi(list_st[index])
        isi_dict[key].append(isi)
    bins = np.arange(0,0.3, 0.01)
    for index, key in enumerate(isi_dict.keys()):
        ax.hist(isi_dict[key], bins,alpha=1, label=key, histtype='step')
    # ax.legend(loc='upper right', fontsize=fontsize-4)
    ax.set_xlabel('ISI (s)', fontsize=fontsize)
    ax.set_ylabel('Count', fontsize=fontsize)
    ax.set_title('Single unit ISI', fontsize=fontsize)
    return ax


def get_cv2(isis):
    """
    Function calculating the CV2 given a list of ISI. Original formula in
    Van Vreijsvik et al. 2010

    Parameters
    ----------
    isis: list
        list of interspike intervals
    """
    cv2 = np.sum([2*np.sum(np.abs(trial_isi[:-1]-trial_isi[1:]) / (trial_isi[:-1]+trial_isi[1:])) for trial_isi in isis]
                )/np.sum([len(trial_isi)-1 if len(trial_isi) > 0 else 0 for trial_isi in isis])
    return cv2


def plot_cv2(ax, sts, gamma, ppd, sep, fontsize):
    """
    Plot representing the CV2 of one neuron of one
    dataset, for the original data, the gamma and the ppd data.

    Parameters
    ----------
    ax: matplotlib.pyplot.axes
        ax where to plot the figure
    sts: list
        list of neo spiketrains of the original data
    gamma: list
        list of neo spiketrains of the gamma data
    ppd: list
        list of neo spiketrains of the PPD data
    neuron: int
        index of the neuron being plotted
    fontsize: int
        fontsize of the legend
    """
    processes = {'original': sts, 'gamma': gamma, 'ppd': ppd}
    # loop over the neurons
    cv2_dict = {'original': [], 'gamma': [], 'ppd': []}

    # original
    cv2_list = []
    # for neuron, conc_st in enumerate(sts_list):
    for neuron, conc_st in enumerate(sts):
        trial_list = create_st_list(conc_st, sep=sep)
        isis = [np.diff(st.magnitude)
                for st in trial_list
                if len(st) > 1]
        cv2 = get_cv2(isis)
        cv2_list.append(cv2)
    cv2_array = np.array(cv2_list)[~np.isnan(np.array(cv2_list))]
    cv2_dict['original'].append(cv2_array)

    # gamma
    cv2_list = []
    for neuron, conc_st in enumerate(gamma):
        trial_list = create_st_list(conc_st, sep=sep)
        isis = [np.diff(st.magnitude)
                for st in trial_list
                if len(st) > 1]
        cv2 = get_cv2(isis)
        cv2_list.append(cv2)
    cv2_array = np.array(cv2_list)[~np.isnan(np.array(cv2_list))]
    cv2_dict['gamma'].append(cv2_array)

    # ppd
    cv2_list = []
    for neuron, conc_st in enumerate(ppd):
        trial_list = create_st_list(conc_st, sep=sep)
        isis = [np.diff(st.magnitude)
                for st in trial_list
                if len(st) > 1]
        cv2 = get_cv2(isis)
        cv2_list.append(cv2)
    cv2_array = np.array(cv2_list)[~np.isnan(np.array(cv2_list))]
    cv2_dict['ppd'].append(cv2_array)

    bins = np.arange(0, 1.5, 0.1)
    for key in cv2_dict.keys():
        ax.hist(cv2_dict[key], bins, alpha=1, label=key, histtype='step')
    # ax.legend(loc='upper right', fontsize=fontsize-4)
    ax.set_title('CV2 of all units', fontsize=fontsize)
    ax.set_ylabel('Count', fontsize=fontsize)
    ax.set_xlabel('CV2', fontsize=fontsize)
    return ax


def panelA_plot(axes, sts, gamma, ppd, neuron, max_refractory, sep,
                  epoch_length, sampling_period, fontsize):
    """
    Function representing Panel A of figure 8 of the manuscript.
    Comparison of statistics of the original to the generated artificial data
    during the movement epoch (PGHF trial type). In blue, orange and green we
    represent original, PPD and gamma data spike trains respectively.
    Left graph: average firing rate of a single unit across one epoch of500ms;
    second from left: ISI distribution of a single unit; third from left:
    averageSG: averaged over what? AS: CV2 is estimated trial wise and then
    averaged CV2 estimated trial wise for all neurons; fourth from left:
    dead time as minimum ISI for all neurons.

    Parameters
    ----------
    ax: matplotlib.pyplot.axes
        ax where to plot the figure
    sts: list
        list of neo spiketrains of the original data
    gamma: list
        list of neo spiketrains of the gamma data
    ppd: list
        list of neo spiketrains of the PPD data
    neuron: int
        index of the neuron being plotted
    max_refractory: quantity
        maximal refractory period
    sep: quantity
        separation time within trials
    epoch_length: quantity
        trial duration (typically 500pq.ms)
    sampling_period: quantity
        sampling period of the recording (30.000Hz)
    fontsize: int
        fontsize of the legend
    """
    (ax1, ax2, ax3, ax4) = axes

    # second plot
    ax1 = plot_trial_firing_rate(ax1,
                                 sts=sts,
                                 gamma=gamma,
                                 ppd=ppd,
                                 neuron=neuron,
                                 max_refractory=max_refractory,
                                 sep=sep,
                                 epoch_length=epoch_length,
                                 sampling_period=sampling_period,
                                 fontsize=fontsize)
    ax1.set_ylim([0,50])

    # isi distribution plot
    ax2 = plot_isi(ax=ax2,
                   sts=sts,
                   gamma=gamma,
                   ppd=ppd,
                   neuron=neuron,
                   fontsize=fontsize)

    # third plot
    ax3 = plot_cv2(ax=ax3,
                   sts=sts,
                   gamma=gamma,
                   ppd=ppd,
                   sep=sep,
                   fontsize=fontsize)

    # fourth plot
    ax4 = plot_rp(ax=ax4,
                  sts=sts,
                  gamma=gamma,
                  ppd=ppd,
                  max_refractory=max_refractory,
                  sampling_period=sampling_period,
                  sep=sep,
                  fontsize=fontsize)

    return axes

def estimate_rate(spiketrain, binsize,
                  epoch_length, winlen):
    """
    Function estimating average rate for a neuron in a concatenated spike train

    Parameters
    ----------
    spiketrain: neo.Spiketrain
        spiketrain whose firing rate is estimated
    binsize: quantity
        binsize of the spade analysis
    epoch_length: quantity
        length of epoch segmentation within one trial
    winlen: int
        window length of the spade analysis

    Returns
    -------
    rate: float
        average firing rate of the neuron, calculated as spike count over
        time (disregarding the padding time in between trials)
    """
    sep = 2 * binsize * winlen
    epoch_length = epoch_length.rescale(pq.s)
    sep = sep.rescale(pq.s)
    number_of_trials = int(spiketrain.t_stop.rescale(pq.s) / (
            epoch_length + sep))
    spike_count = len(spiketrain)
    rate = spike_count / (number_of_trials * epoch_length)
    return rate


def calculate_fps(surrogates_ppd, surrogates_gamma, sessions):
    """
    Function calculating the number of false positives across all analyzed
    datasets and sessions, separately for ppd and gamma process.
    The function loads the results and counts the number of patterns
    detected by SPADE.

    sessions: list
        list of strings corresponding to the analyzed sessions
    surrogates_ppd: list
        list of name tags of folders of surrogates for ppd
        #TODO: to change
    surrogates_gamma: list
        list of name tags of folders of surrogates for gamma
        #TODO: to change

    Returns
    -------
    ppd_fps, gamma_fps: dict
        dictionary of number of false positives for ppd and gamma.
        Each key corresponds to a surrogate method

    """
    gamma_fps = {}
    ppd_fps = {}
    for surrogate in surrogates_ppd:
        ppd_fps[surrogate] = 0
        for session in sessions:
            folder_res = [f.path for f in
                          os.scandir('../results/artificial_data/'+
                                     surrogate+'/ppd/'+ session +'/'
                                     ) if f.is_dir()]
            for result in folder_res:
                res = np.load(result + '/filtered_res.npy', allow_pickle=True)
                if len(res[0]) > 0:
                    ppd_fps[surrogate] += len(res[0])
    for index, surrogate in enumerate(surrogates_gamma):
        gamma_fps[surrogates_ppd[index]] = 0
        for session in sessions:
            folder_res = [f.path for f in
                          os.scandir('../results/artificial_data/'+
                                     surrogate+'/gamma/'+ session +'/'
                                     ) if f.is_dir()]
            for result in folder_res:
                res = np.load(result + '/filtered_res.npy', allow_pickle=True)
                if len(res[0]) > 0:
                    gamma_fps[surrogates_ppd[index]] += len(res[0])
    return ppd_fps, gamma_fps


def calculate_fps_rate(surrogates_ppd, surrogates_gamma,
                       sessions, binsize, winlen, epoch_length):
    """
    Function calculating the average rate of the neurons involved in patterns
    detected in the artificial data, separately for gamma and ppd data.
    sessions: list
        list of strings corresponding to the analyzed sessions
    surrogates_ppd: list
        list of name tags of folders of surrogates for ppd
        #TODO: to change
    surrogates_gamma: list
        list of name tags of folders of surrogates for gamma
        #TODO: to change

    Returns
    -------
    neurons_fr: dict
        dictionary containing for gamma and ppd process the average rate of
        neurons involved in patterns. neurons_fr[process] is a dictionary
        itself, having keys as the surrogate methods.
    """

    neurons_fr = {'gamma':{}, 'ppd':{}}

    # loading original data
    for surrogate in surrogates_ppd:
        neurons_fr['ppd'][surrogate] = []
        for session in sessions:
            folder_res = [f.path for f in os.scandir(
                '../results/artificial_data/' + surrogate + '/ppd/' + session + '/') if
                          f.is_dir()]
            for result in folder_res:
                res = np.load(result + '/filtered_res.npy', allow_pickle=True)
                for pattern in res[0]:
                    for neuron in pattern['neurons']:
                        # loading original data and retrieve the rate
                        behavioral_context = result.split('/')[-1]
                        original_data_path = '../data/concatenated_spiketrains/' + \
                                             session + '/' + behavioral_context + \
                                             '.npy'
                        original_sts = np.load(original_data_path,
                                               allow_pickle=True)
                        rate = estimate_rate(original_sts[neuron],
                                             binsize=binsize,
                                             winlen=winlen,
                                             epoch_length=epoch_length)
                        neurons_fr['ppd'][surrogate].append(rate.magnitude)
        neurons_fr['ppd'][surrogate] = \
            np.array(neurons_fr['ppd'][surrogate]).flatten()

    for surrogate in surrogates_gamma:
        neurons_fr['gamma'][surrogate] = []
        for session in sessions:
            folder_res = [f.path for f in os.scandir(
                '../results/artificial_data/' + surrogate + '/gamma/' + session + '/') if
                          f.is_dir()]
            for result in folder_res:
                res = np.load(result + '/filtered_res.npy', allow_pickle=True)
                for pattern in res[0]:
                    for neuron in pattern['neurons']:
                        # loading original data and retrieve the rate
                        behavioral_context = result.split('/')[-1]
                        original_data_path = '../data/concatenated_spiketrains/' + \
                                             session + '/' + behavioral_context + \
                                             '.npy'
                        original_sts = np.load(original_data_path,
                                               allow_pickle=True)
                        rate = \
                        estimate_rate(original_sts[neuron],
                                             binsize=binsize,
                                             winlen=winlen,
                                             epoch_length=epoch_length)
                        neurons_fr['gamma'][surrogate].append(rate.magnitude)
        neurons_fr['gamma'][surrogate] = \
            np.array(neurons_fr['gamma'][surrogate]).flatten()

    return neurons_fr


def plot_inset_fps_fr(ax_fps_fr, process, sessions, surrogates_ppd,
                      surrogates_gamma, surrogates_tag, binsize,
                      winlen, epoch_length, scale):
    """
    Function producing panel C of fig 8 of the paper.
    It calculates the distribution of average firing rates of neurons
    participating in FPs across surrogate techniques (in different colors),
    left for PPD and right for gamma process data analyses.

    Parameters
    ----------
    ax_fps_fr: matplotlib.pyplot.axes
        axes where to plot the inset
    process: str
        strings of the point process models employed (e.g. 'ppd' or 'gamma')
    sessions: list
        list of strings corresponding to the analyzed sessions
    surrogates_ppd: list
        list of name tags of folders of surrogates for ppd
        #TODO: to change
    surrogates_gamma: list
        list of name tags of folders of surrogates for gamma
        #TODO: to change
    surrogates_tag: list
        list of strings for names of surrogate techniques
    binsize: quantity
        binsize of the spade analysis
    epoch_length: quantity
        length of epoch segmentation within one trial
    winlen: int
        window length of the spade analysis
    scale: int
        scale of the firing rate (binning of the histogram)
    """
    bins = np.arange(0, 80, scale)
    neurons_fr = calculate_fps_rate(surrogates_ppd=surrogates_ppd,
                                    surrogates_gamma=surrogates_gamma,
                                    sessions=sessions,
                                    binsize=binsize,
                                    winlen=winlen,
                                    epoch_length=epoch_length)
    if process == 'gamma':
        for index, surrogate in enumerate(surrogates_gamma):
            hist = np.histogram(neurons_fr['gamma'][surrogate], bins=bins,
                                density=True)[0]
            ax_fps_fr.plot(hist, label=surrogates_tag[index])
            ax_fps_fr.set_xticklabels(
                [x * scale for x in ax_fps_fr.get_xticks()])
            ax_fps_fr.legend()
    if process == 'ppd':
        for index, surrogate in enumerate(surrogates_ppd):
            hist = np.histogram(neurons_fr['ppd'][surrogate], bins=bins,
                                density=True)[0]
            ax_fps_fr.plot(hist, label=surrogates_tag[index])
            ax_fps_fr.set_xticklabels(
                [x * scale for x in ax_fps_fr.get_xticks()])
    return ax_fps_fr


def plot_inset(ax_num_fps, index, process, sessions, surrogates_ppd,
               surrogates_gamma, label_size, tick_size):
    """
    Function producing the inset left or right of fig 8 of the paper.
    It calculates the number of false positives detected across surrogate
    techniques in all datasets, either for PPD model or for gamma model.

    Parameters
    ----------
    ax_num_fps: matplotlib.pyplot.axes
        axes where to plot the inset
    index: int
        index respective to the position of the results for the plotted process
        0 for left, 1 for right
    process: str
        strings of the point process models employed (e.g. 'ppd' or 'gamma')
    sessions: list
        list of strings corresponding to the analyzed sessions
    surrogates_ppd: list
        list of name tags of folders of surrogates for ppd
        #TODO: to change
    surrogates_gamma: list
        list of name tags of folders of surrogates for gamma
        #TODO: to change
    label_size: int
        label size for title
    tick_size: int
        tick size for x and y ticks
    """
    for index_surr, surrogate in enumerate(surrogates_ppd):
        fps = calculate_fps(sessions=sessions,
                            surrogates_ppd=surrogates_ppd,
                            surrogates_gamma=surrogates_gamma)[index]
        if process == 'ppd':
            ax_num_fps.bar(index_surr + 1,
                           fps[surrogate],
                           width=0.5, color='grey', label=surrogate)
        elif process == 'gamma':
            ax_num_fps.bar(index_surr + 1,
                           fps[surrogates_ppd[index_surr]],
                           width=0.5, color='grey', label=surrogate)
        else:
            raise ImportError('process not recognized')
        ax_num_fps.set_ylabel('FPs', size=label_size)
    ax_num_fps.set_ylim([0, 95])
    ax_num_fps.set_xticks(range(1, len(surrogates_ppd) + 1))
    ax_num_fps.tick_params(axis='both', which='major',
                           labelsize=tick_size)
    return ax_num_fps


def plot_fps(axes, processes, sessions, surrogates_ppd, surrogates_gamma):
    """
    Function producing the figure of panel B of fig 8 of the paper.
    It calculates the number of false positives detected across surrogate
    techniques in all datasets, left for PPD model, and right for gamma model.

    Parameters
    ----------
    processes: list
        list of strings of the point process models employed (e.g. ['ppd',
        'gamma'])
    sessions: list
        list of strings corresponding to the analyzed sessions
    surrogates_ppd: list
        list of name tags of folders of surrogates for ppd
        #TODO: to change
    surrogates_gamma: list
        list of name tags of folders of surrogates for gamma
        #TODO: to change
    """
    # Plotting parameters
    label_size = 24
    title_size = 26
    tick_size = 24
    surrogates_tag = ['UD', 'Bin-Shuff', 'TR-Shift', 'UD-DT', 'J-ISI-D',
                      'ISI-D']
    for index, process in enumerate(processes):
        if process == 'ppd':
            ax_num_fps = plot_inset(axes[index], index, process, sessions,
                                    surrogates_ppd, surrogates_gamma,
                                    label_size, surrogates_tag, tick_size)
            ax_num_fps.set_xticks([])
            ax_num_fps.set_xticklabels('')
            ax_num_fps.set_title(
                'FPs detected in artificial data \n across '
                'surrogate techniques',
                size=title_size)
        if process == 'gamma':
            ax_num_fps = plot_inset(axes[index], index, process, sessions,
                                    surrogates_ppd, surrogates_gamma,
                                    label_size, surrogates_tag, tick_size)
            ax_num_fps.set_xticklabels(surrogates_tag, rotation=45,
                                       size=tick_size)

    return axes


def figure8_artificial_data(sts, gamma, ppd, neuron, max_refractory,
                            sep, sampling_period, epoch_length,
                            sessions, surrogates_ppd, surrogates_gamma,
                            surrogates_tag,
                            processes,
                            label_size, tick_size, scale):
    """
    Function reproducing figure 8 of the paper.
    Evaluation and analysis of false positive for pattern detection with SPADE.
    Panel A: Comparison of statistics of the original to the generated
    artificial data during the movement epoch (PGHF trial type). In blue,
    orange and green we represent original, PPD and gamma data spike trains
    respectively. Left graph: average firing rate of a single unit across one
    epoch of500ms; second from left: ISI distribution of a single unit;
    third from left: average CV2 estimated trial wise for all neurons;
    fourth from left: dead time as minimum ISI for all neurons. Panel B:
    number of false positives (FPs) detected across surrogate techniques and
    all48 (2(sessions)\times6(epochs)\times4(trialtypes)) data sets
    analyzed, left for PPD and right for gamma process data analyses. Panel C:
    distribution of average firing rates of neurons participating in FPs
    across surrogate techniques (in different colors), left for PPD and right
    for gamma process data analyses.

    Parameters
    ----------
    sts: list
        list of neo spiketrains of the original data
    gamma: list
        list of neo spiketrains of the gamma data
    ppd: list
        list of neo spiketrains of the PPD data
    neuron: int
        neuron chosen to represent its firing rate modulation and ISI distr.
    max_refractory: quantity
        maximal refractory period
    sep: quantity
        separation time within trials
    epoch_length: quantity
        trial duration (typically 500pq.ms)
    sampling_period: quantity
        sampling period of the recording (30.000Hz)
    sessions: list
        list of sessions analyzed
    surrogates_ppd: list
        list of name tags of folders of surrogates for ppd
        #TODO: to change
    surrogates_gamma: list
        list of name tags of folders of surrogates for gamma
        #TODO: to change
    processes: list
        list of strings of the point process models employed (e.g. ['ppd',
        'gamma'])
    label_size: int
        label size for legend
    tick_size: int
        tick size for y and x axes
    scale: int
        scale of the firing rate (binning of the histogram)
    """
    # gridspec inside gridspec
    fig = plt.figure(figsize=(7.5, 8.75))
    params = {'legend.fontsize': 8,
              'figure.figsize': (7.5, 8.75),
              'axes.labelsize': 10,
              'axes.titlesize': 10,
              'xtick.labelsize': 8,
              'ytick.labelsize': 8}
    pylab.rcParams.update(params)

    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[0.3, 0.7])

    # Panel A
    gs0 = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=4,
                                           subplot_spec=gs[0], wspace=0.5)
    ax01 = fig.add_subplot(gs0[0])
    ax02 = fig.add_subplot(gs0[1])
    ax03 = fig.add_subplot(gs0[2])
    ax04 = fig.add_subplot(gs0[3])
    axes = (ax01, ax02, ax03, ax04)
    panelA_plot(axes=axes, sts=sts, gamma=gamma, ppd=ppd, neuron=neuron,
                max_refractory=max_refractory, sep=sep, fontsize=8,
                sampling_period=sampling_period, epoch_length=epoch_length)
    ax03.set_xlim(0, 2)
    ax02.set_xlim(0, 0.2)
    ax01.set_ylim(0, 30)
    ax04.legend(fontsize=params['legend.fontsize'])
    plt.text(x=-0.45, y=1.05, s='A', transform=ax01.transAxes, fontsize=10)

    # Panel B and C
    gs1 = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=2,
                                           subplot_spec=gs[1])

    # ppd

    gs10 = gridspec.GridSpecFromSubplotSpec(nrows=2, ncols=1,
                                            subplot_spec=gs1[0], hspace=0.3)
    ax11 = fig.add_subplot(gs10[0])
    ax12 = fig.add_subplot(gs10[1])
    gs20 = gridspec.GridSpecFromSubplotSpec(nrows=2, ncols=1,
                                            subplot_spec=gs1[1], hspace=0.3)
    ax21 = fig.add_subplot(gs20[0])
    ax22 = fig.add_subplot(gs20[1])
    # false positives

    for index, process in enumerate(processes):
        if process == 'ppd':
            ax11 = plot_inset(ax_num_fps=ax11,
                              index=index,
                              process=process,
                              sessions=sessions,
                              surrogates_ppd=surrogates_ppd,
                              surrogates_gamma=surrogates_gamma,
                              label_size=label_size,
                              tick_size=tick_size)
            ax11.set_xticklabels(surrogates_tag, rotation=45,
                                 size=tick_size)
            ax11.set_ylim([0, 100])
            plt.text(x=0.45, y=1.05, s='PPD', transform=ax11.transAxes,
                     fontsize=10)
            plt.text(x=- 0.25, y=1.05, s='B', transform=ax11.transAxes,
                     fontsize=title_size)
            ax12 = plot_inset_fps_fr(ax_fps_fr=ax12,
                                              process=process,
                                              sessions=sessions,
                                              surrogates_ppd=surrogates_ppd,
                                              surrogates_gamma=surrogates_gamma,
                                              surrogates_tag=surrogates_tag,
                                              scale=scale,
                                              binsize=binsize,
                                              winlen=winlen,
                                              epoch_length=epoch_length)
            ax12.set_ylabel('distr. FPs by neuronal FR')
            ax12.set_xlabel('firing rate (Hz)')
            ax12.set_ylim([0, 0.15])
            plt.text(x=-0.25, y=1.05, s='C', transform=ax12.transAxes,
                     fontsize=title_size)
        if process == 'gamma':
            ax21 = plot_inset(ax_num_fps=ax21,
                                       index=index,
                                       process=process,
                                       sessions=sessions,
                                       surrogates_ppd=surrogates_ppd,
                                       surrogates_gamma=surrogates_gamma,
                                       label_size=label_size,
                                       tick_size=tick_size)
            ax21.set_xticklabels(surrogates_tag, rotation=45,
                                 size=tick_size)
            ax21.set_ylabel('')
            ax21.set_ylim([0, 100])
            plt.text(x=0.45, y=1.05, s='Gamma', transform=ax21.transAxes,
                     fontsize=title_size)
            ax22 = plot_inset_fps_fr(ax_fps_fr=ax22,
                                              process=process,
                                              sessions=sessions,
                                              surrogates_ppd=surrogates_ppd,
                                              surrogates_gamma=surrogates_gamma,
                                              surrogates_tag=surrogates_tag,
                                              scale=scale,
                                              binsize=binsize,
                                              winlen=winlen,
                                              epoch_length=epoch_length)
            ax22.legend()
            ax22.set_ylim([0, 0.15])
            ax22.set_xlabel('firing rate (Hz)')
    # plt.savefig('/home/stella/Documents/Juelich/projects/'
    #             'surrogates_comparison/manuscript/figures/'
    #             '/fig_artificial_data.png')
    # plt.savefig('/home/stella/Documents/Juelich/projects/'
    #             'surrogates_comparison/manuscript/figures/'
    #             '/fig_artificial_data.eps')
    plt.show()


if __name__ == "__main__":
    # Define parameters
    binsize = 5 * pq.ms
    winlen = 12
    sep = 2 * winlen * binsize
    max_refractory = 4 * pq.ms
    epoch_length = 500 * pq.ms
    sampling_period = 0.1 * pq.ms
    neuron = 39
    label_size = 10
    title_size = 10
    tick_size = 8

    # bins for last panel
    scale = 3

    # loading of data for panel A
    session = 'i140703-001'
    epoch = 'movement'
    trialtype = 'PGHF'
    sts = np.load(f'../data/concatenated_spiketrains/{session}/'
                  f'{epoch}_{trialtype}.npy', allow_pickle=True)
    gamma = np.load(f'../data/artificial_data/gamma/{session}/'
                    f'gamma_{epoch}_{trialtype}.npy', allow_pickle=True)
    ppd = np.load(f'../data/artificial_data/ppd/{session}/'
                  f'ppd_{epoch}_{trialtype}.npy', allow_pickle=True)

    surrogates_gamma = ['ud', 'bin_shuffling',
                        'shift_st', 'udrp',
                        'jisi', 'isi']
    surrogates_ppd = ['ud', 'bin_shuffling', 'shift_st', 'udrp', 'jisi', 'isi']
    sessions = ['i140703-001', 'l101210-001']
    processes = ['ppd', 'gamma']

    surrogates_tag = ['UD', 'Bin-Shuff', 'TR-Shift', 'UD-DT', 'J-ISI-D',
                      'ISI-D']
    figure8_artificial_data(sts=sts,
                            gamma=gamma,
                            ppd=ppd,
                            neuron=neuron,
                            max_refractory=max_refractory,
                            sep=sep,
                            sampling_period=sampling_period,
                            epoch_length=epoch_length,
                            sessions=sessions,
                            surrogates_ppd=surrogates_ppd,
                            surrogates_gamma=surrogates_gamma,
                            surrogates_tag=surrogates_tag,
                            processes=processes,
                            label_size=label_size,
                            tick_size=tick_size,
                            scale=scale)