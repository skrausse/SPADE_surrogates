import numpy as np
import matplotlib.pyplot as plt
import quantities as pq

import fig7_surrogate_statistics_config as cfg

DATA_PATH = cfg.DATA_PATH
PLOT_PATH = cfg.PLOT_PATH

SURR_METHODS = cfg.SURR_METHODS
DATA_TYPES = cfg.DATA_TYPES

XLABELPAD = cfg.XLABELPAD
YLABELPAD = cfg.YLABELPAD
ORIGINAL_LINEWIDTH = cfg.ORIGINAL_LINEWIDTH

LABELS = cfg.LABELS
COLORS = cfg.COLORS

LINE_STYLES = cfg.LINE_STYLES

LETTERS = cfg.LETTERS


def plot_clipped_firing_rate_and_eff_moved(axes_clip, axes_sync=None):
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
    results = np.load(
        f'{DATA_PATH}/clipped_rates.npy', allow_pickle=True).item()
    rates = results['rates']
    ratio_clipped = results['ratio_clipped']
    ratio_clipped_surr = results['ratio_clipped_surr']

    ratio_indep_moved = results['ratio_indep_moved']
    ratio_moved = results['ratio_moved']

    for type_id, data_type in enumerate(DATA_TYPES):
        axes_clip[type_id].plot(
            rates[type_id], 1. - np.array(ratio_clipped[type_id]),
            label='original',
            linewidth=ORIGINAL_LINEWIDTH,
            color=COLORS['original'],
            linestyle=LINE_STYLES['original']
        )
        if axes_sync is not None:
            axes_sync[type_id].plot(
                rates[type_id], ratio_indep_moved[type_id],
                label='indep.',
                linewidth=ORIGINAL_LINEWIDTH,
                linestyle=LINE_STYLES['original']
            )
        for surr_method in SURR_METHODS:
            axes_clip[type_id].plot(
                rates[type_id],
                1. - np.array(ratio_clipped_surr[type_id][surr_method]),
                label=surr_method,
                color=COLORS[surr_method],
                linestyle=LINE_STYLES[surr_method])
            if axes_sync is not None:
                axes_sync[type_id].plot(
                    rates[type_id],
                    ratio_moved[type_id][surr_method],
                    label=surr_method,
                    color=COLORS[surr_method],
                    linestyle=LINE_STYLES[surr_method])

        axes_clip[type_id].set_xlabel(r'$\lambda$ in Hz',
                                      labelpad=XLABELPAD)

        if axes_sync is not None:
            axes_sync[type_id].set_xlabel(r'$\lambda$ in Hz',
                                          labelpad=XLABELPAD)


def _plot_isi(axes_isi, data_type, type_id, surr_method):
    results = np.load(
        f'{DATA_PATH}/isi_{data_type}_{surr_method}.npy',
        allow_pickle=True).item()

    bin_edges = results['bin_edges']
    hist = results['hist']

    linewidth = ORIGINAL_LINEWIDTH if surr_method == 'original' else 1.

    label = LABELS[surr_method] if type_id == 0 else None

    axes_isi[type_id].plot(
        bin_edges[:-1] + bin_edges[0] / 2,
        hist,
        linewidth=linewidth,
        label=label,
        color=COLORS[surr_method],
        linestyle=LINE_STYLES[surr_method])


def _plot_ac_cc(axes_ac, axes_cc, type_id, data_type,
                surr_method):
    linewidth = ORIGINAL_LINEWIDTH if surr_method == 'original' else 1.
    for axes, corr_type in zip((axes_ac, axes_cc), ('ac', 'cc')):
        results = np.load(
            f'{DATA_PATH}/{corr_type}_{data_type}_{surr_method}.npy',
            allow_pickle=True).item()

        hist_times = results['hist_times']
        hist = results['hist']

        axes[type_id].plot(
            hist_times,
            hist,
            label=surr_method, linewidth=linewidth, color=COLORS[surr_method],
            linestyle=LINE_STYLES[surr_method])


def _label_axes(axes_isi, axes_ac, axes_cc,
                dither, rate):
    for type_id, data_type in enumerate(DATA_TYPES):
        for analysis_id, axis in enumerate(
                (axes_isi[type_id], axes_cc[type_id])):
            axis.set_xlabel(r'$\tau$ in ms', labelpad=XLABELPAD)

        axes_ac[type_id].set_ylim(bottom=0.8 * rate,
                                  top=1.1 * rate)
        axes_cc[type_id].set_ylim(bottom=0.8 * rate,
                                  top=1.6 * rate)
        axes_ac[type_id].set_xlim(left=-3.05 * dither, right=3.05 * dither)

        axes_cc[type_id].set_xlim(left=-3.05 * dither, right=3.05 * dither)


def plot_statistical_analysis_of_single_rate(
        axes_isi, axes_ac, axes_cc, dither=15. * pq.ms,
        rate=50 * pq.Hz):
    for type_id, data_type in enumerate(DATA_TYPES):
        _plot_isi(axes_isi, data_type, type_id, surr_method='original')
        _plot_ac_cc(axes_ac, axes_cc, type_id, data_type,
                    surr_method='original')

        for surr_method in SURR_METHODS:
            _plot_isi(axes_isi, data_type, type_id, surr_method)

            _plot_ac_cc(axes_ac, axes_cc, type_id, data_type,
                        surr_method)

    _label_axes(axes_isi, axes_ac, axes_cc,
                dither, rate)


def plot_firing_rate_change(axis, data_type='Gamma'):
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
    results = np.load(f'{DATA_PATH}/rate_step.npy', allow_pickle=True).item()

    rate_original = results[data_type]['original']
    axis.plot(rate_original.times, rate_original.simplified.magnitude,
              label='original', linewidth=ORIGINAL_LINEWIDTH,
              color=COLORS['original'], linestyle=LINE_STYLES['original'])

    for surr_method in SURR_METHODS:
        rate_dithered = results[data_type][surr_method]

        axis.plot(rate_dithered.times, rate_dithered.simplified.magnitude,
                  label=surr_method, color=COLORS[surr_method],
                  linestyle=LINE_STYLES[surr_method])

    axis.set_xlabel(r't in ms', labelpad=XLABELPAD)
    axis.set_ylabel(r'$\lambda(t)$ in Hz', labelpad=YLABELPAD)

    axis.set_xticks([0., 50., 100., 150.])


def plot_eff_moved(axis, data_type='Gamma'):
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
    results = np.load(
        f'{DATA_PATH}/clipped_rates.npy', allow_pickle=True).item()

    rates = results['rates']
    ratio_indep_moved = results['ratio_indep_moved']
    ratio_moved = results['ratio_moved']

    for type_id, dt_type in enumerate(DATA_TYPES):
        if dt_type == data_type:
            axis.plot(
                rates[type_id], ratio_indep_moved[type_id],
                label='indep.',
                linewidth=ORIGINAL_LINEWIDTH,
                color=COLORS['original'],
                linestyle=LINE_STYLES['original'])
            for surr_method in SURR_METHODS:
                axis.plot(
                    rates[type_id],
                    ratio_moved[type_id][surr_method],
                    label=surr_method,
                    color=COLORS[surr_method],
                    linestyle=LINE_STYLES[surr_method])

    axis.set_xlabel(r'$\lambda$ in Hz', labelpad=XLABELPAD)
    axis.set_ylabel(r'$N_{moved}/N$', labelpad=YLABELPAD)


def plot_cv_change(axis):
    results = np.load(f'{DATA_PATH}/cv_change.npy', allow_pickle=True).item()

    cvs_real = results['cvs_real']
    axis.plot(cvs_real, cvs_real, linewidth=ORIGINAL_LINEWIDTH,
              color=COLORS['original'], linestyle=LINE_STYLES['original'])

    for surr_method in SURR_METHODS:
        cvs_dithered = results[surr_method]
        axis.plot(cvs_real, cvs_dithered, label=surr_method,
                  color=COLORS[surr_method],
                  linestyle=LINE_STYLES[surr_method])
    axis.set_xlabel('CV - original', labelpad=XLABELPAD)
    axis.set_ylabel('CV - dithered', labelpad=YLABELPAD)


def _hide_x_ticks(axis):
    axis.set_xticklabels([])
    axis.tick_params(bottom=False)


def _hide_y_ticks(axis):
    axis.set_yticklabels([])
    axis.tick_params(left=False)


def _dash_spines(axis):
    for direction in ('left', 'right', 'bottom', 'top'):
        axis.spines[direction].set_linewidth(.5)
        axis.spines[direction].set_linestyle((0, (8, 8)))


def plot_statistics_overview():
    plt.rcParams.update({'font.size': 10,
                         'text.usetex': True,
                         'lines.linewidth': 0.75,
                         'figure.figsize': (7.5, 8.75)})
    start_down = 0.05
    height_figure = 0.12
    # epsilon = 0.035
    epsilon = 0.05
    delta_down = 0.05 + height_figure + epsilon
    delta_up = 0.05
    epsilons = [2*epsilon, epsilon, 0, 0]
    eta = 0.025
    left_move = -0.085

    fig = plt.figure()

    axes = [[
        fig.add_axes(
            rect=[left_move + delta_down + (1-(delta_down+delta_up))/3*x,
                  delta_down + (1-(delta_down+delta_up)-2*epsilon)/4*(3-y)
                  + epsilons[y],
                  (1-(delta_down+delta_up))/3,
                  (1-(delta_down+delta_up)-2*epsilon)/4])
        for x in range(3)] for y in range(4)]

    # for axis in axes:
    #    for ax in axis:
    #        ax.grid()
    #        _dash_spines(ax)

    lower_axes = \
        [fig.add_axes(
            rect=[left_move + delta_down
                  + (1-(delta_down+delta_up + 2*epsilon))/3*x + epsilon * x
                  + eta/2*x,
                  start_down,
                  (1-(delta_down+delta_up + 2*epsilon))/3 - eta,
                  height_figure])
         for x in range(3)]

    # for ax in lower_axes:
    #    ax.grid()
    #    _dash_spines(ax)

    # [left, bottom, width, height]

    # gs = fig.add_gridspec(4, 3, hspace=0, wspace=0)
    # axes = gs.subplots(sharey='row')
    axes_clip, axes_isi, axes_ac, axes_cc = axes
    axis_step, axis_moved, axis_cv = lower_axes

    axes_clip[0].set_ylabel(r'$1 - N_{clip}/N$', labelpad=YLABELPAD)
    axes_isi[0].set_ylabel(r'$p(\tau)$ in 1/s', labelpad=YLABELPAD)
    axes_ac[0].set_ylabel('ACH in 1/s', labelpad=YLABELPAD)
    axes_cc[0].set_ylabel('CCH in 1/s', labelpad=YLABELPAD)

    plot_clipped_firing_rate_and_eff_moved(axes_clip)

    plot_statistical_analysis_of_single_rate(
        axes_isi, axes_ac, axes_cc, rate=60*pq.Hz)

    plot_firing_rate_change(axis_step)
    plot_eff_moved(axis_moved)
    plot_cv_change(axis_cv)

    for data_id, data_type in enumerate(DATA_TYPES):
        subtitle = data_type
        axes[0][data_id].set_title(subtitle)

    for axis_id, axis in enumerate(axes):
        if axis_id < 3:
            axis[0].text(
                -0.2, 1.05, LETTERS[axis_id],
                transform=axis[0].transAxes, fontsize=15)
        _hide_y_ticks(axis[1])
        _hide_y_ticks(axis[2])

    for axis_id, axis in enumerate(lower_axes):
        axis.text(
            -0.2, 1.05, LETTERS[3 + axis_id],
            transform=axis.transAxes, fontsize=15)

    for axis in axes_ac:
        _hide_x_ticks(axis)

    for axis in axes:
        ylims = [axis[x].get_ylim() for x in range(3)]
        ylim = (min(ylims[x][0] for x in range(3)),
                max(ylims[x][1] for x in range(3)))
        [axis[x].set_ylim(ylim) for x in range(3)]

    # fig.suptitle('Surrogate Statistics', y=0.96)
    handles, labels = axes_isi[0].get_legend_handles_labels()
    x, y = 2, 0
    legend_loc = (
        left_move + delta_down + (1-(delta_down+delta_up))/3*x
        + (1-(delta_down+delta_up))/3 + 0.003,
        delta_down + (1-(delta_down+delta_up)-2*epsilon)/4*(3-y)
        + epsilons[y] + 0.02)  # loc=(x, y)
    legend = fig.legend(handles, labels, fontsize=8., loc=legend_loc)
    frame = legend.get_frame()
    frame.set_facecolor('0.9')
    frame.set_edgecolor('0.9')

    fig.savefig(f'{PLOT_PATH}/'
                f'Fig_surrogate_statistics.eps',
                dpi=300)


if __name__ == '__main__':
    plot_statistics_overview()
