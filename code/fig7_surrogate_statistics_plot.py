import numpy as np
import matplotlib.pyplot as plt
import quantities as pq

import fig7_surrogate_statistics_config as cf

DATA_PATH = cf.DATA_PATH
PLOT_PATH = cf.PLOT_PATH

FIG_NAME = cf.FIG_NAME

SURR_METHODS = cf.SURR_METHODS
DATA_TYPES = cf.DATA_TYPES

STEP_DATA_TYPE = cf.STEP_DATA_TYPE

FIRING_RATE = cf.FIRING_RATE
RATES = cf.RATES

DURATION_RATES_STEP = cf.DURATION_RATES_STEP

DITHER = cf.DITHER

FIGSIZE = cf.FIGSIZE
XLABELPAD = cf.XLABELPAD
YLABELPAD = cf.YLABELPAD
ORIGINAL_LINEWIDTH = cf.ORIGINAL_LINEWIDTH
SURROGATES_LINEWIDTH = cf.SURROGATES_LINEWIDTH
FONTSIZE = cf.FONTSIZE

LABELS = cf.LABELS
COLORS = cf.COLORS

LINE_STYLES = cf.LINE_STYLES

LETTERS = cf.LETTERS

AC_BOTTOM = cf.AC_BOTTOM
AC_TOP = cf.AC_TOP
CC_BOTTOM = cf.CC_BOTTOM
CC_TOP = cf.CC_TOP

AC_CC_XLIM = cf.AC_CC_XLIM


def plot_clipped_firing_rate_and_eff_moved(axes_clip):
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

    for type_id, data_type in enumerate(DATA_TYPES):
        axes_clip[type_id].plot(
            rates[data_type], 1. - np.array(ratio_clipped[data_type]),
            label='original',
            linewidth=ORIGINAL_LINEWIDTH,
            color=COLORS['original'],
            linestyle=LINE_STYLES['original']
        )
        for surr_method in SURR_METHODS:
            axes_clip[type_id].plot(
                rates[data_type],
                1. - np.array(ratio_clipped_surr[data_type][surr_method]),
                label=surr_method,
                color=COLORS[surr_method],
                linestyle=LINE_STYLES[surr_method])

        axes_clip[type_id].set_xlabel(r'$\lambda$ in Hz',
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


def _label_axes(axes_isi, axes_ac, axes_cc):
    for axis_isi, axis_cc in zip(axes_isi, axes_cc):
        for axis in (axis_isi, axis_cc):
            axis.set_xlabel(r'$\tau$ in ms', labelpad=XLABELPAD)

    for axis_ac, axis_cc in zip(axes_ac, axes_cc):
        axis_ac.set_ylim(bottom=AC_BOTTOM * FIRING_RATE,
                         top=AC_TOP * FIRING_RATE)
        axis_ac.set_xlim(left=-AC_CC_XLIM * DITHER,
                         right=AC_CC_XLIM * DITHER)

    for axis_cc in axes_cc:
        axis_cc.set_ylim(bottom=CC_BOTTOM * FIRING_RATE,
                         top=CC_TOP * FIRING_RATE)
        axis_cc.set_xlim(left=-AC_CC_XLIM * DITHER,
                         right=AC_CC_XLIM * DITHER)
        axis_cc.set_xticks([-DITHER.magnitude, 0., DITHER.magnitude])


def plot_statistical_analysis_of_single_rate(
        axes_isi, axes_ac, axes_cc):
    for type_id, data_type in enumerate(DATA_TYPES):
        _plot_isi(axes_isi, data_type, type_id, surr_method='original')
        _plot_ac_cc(axes_ac, axes_cc, type_id, data_type,
                    surr_method='original')

        for surr_method in SURR_METHODS:
            _plot_isi(axes_isi, data_type, type_id, surr_method)

            _plot_ac_cc(axes_ac, axes_cc, type_id, data_type,
                        surr_method)

    _label_axes(axes_isi, axes_ac, axes_cc)


def plot_firing_rate_change(axis):
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

    rate_original = results['original']
    axis.plot(rate_original.times, rate_original.simplified.magnitude,
              label='original', linewidth=ORIGINAL_LINEWIDTH,
              color=COLORS['original'], linestyle=LINE_STYLES['original'])

    for surr_method in SURR_METHODS:
        rate_dithered = results[surr_method]

        axis.plot(rate_dithered.times, rate_dithered.simplified.magnitude,
                  label=surr_method, color=COLORS[surr_method],
                  linestyle=LINE_STYLES[surr_method])

    axis.set_xlabel(r't in ms', labelpad=XLABELPAD)
    axis.set_ylabel(r'$\lambda(t)$ in Hz', labelpad=YLABELPAD)

    t_stop = DURATION_RATES_STEP.rescale(pq.ms).magnitude
    axis.set_xticks([0., 1/3 * t_stop, 2/3 * t_stop, t_stop])


def plot_eff_moved(axis):
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

    axis.plot(
        rates[STEP_DATA_TYPE], ratio_indep_moved[STEP_DATA_TYPE],
        label='indep.',
        linewidth=ORIGINAL_LINEWIDTH,
        color=COLORS['original'],
        linestyle=LINE_STYLES['original'])
    for surr_method in SURR_METHODS:
        axis.plot(
            rates[STEP_DATA_TYPE],
            ratio_moved[STEP_DATA_TYPE][surr_method],
            label=surr_method,
            color=COLORS[surr_method],
            linestyle=LINE_STYLES[surr_method])

    axis.set_xlabel(r'$\lambda$ in Hz', labelpad=XLABELPAD)
    axis.set_ylabel(r'$N_{moved}/N$', labelpad=YLABELPAD)

    maximal_rate = RATES[-1]
    axis.set_xticks([1/4 * maximal_rate, 2/4 * maximal_rate,
                     3/4 * maximal_rate, 4/4 * maximal_rate])


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
    axis.set_ylabel('CV - surrogate', labelpad=YLABELPAD)


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
    plt.rcParams.update({'font.size': FONTSIZE,
                         'text.usetex': True,
                         'lines.linewidth': SURROGATES_LINEWIDTH,
                         'figure.figsize': FIGSIZE})
    start_down = 0.05
    height_figure = 0.12
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

    lower_axes = \
        [fig.add_axes(
            rect=[left_move + delta_down
                  + (1-(delta_down+delta_up + 2*epsilon))/3*x + epsilon * x
                  + eta/2*x,
                  start_down,
                  (1-(delta_down+delta_up + 2*epsilon))/3 - eta,
                  height_figure])
         for x in range(3)]

    axes_clip, axes_isi, axes_ac, axes_cc = axes
    axis_step, axis_moved, axis_cv = lower_axes

    axes_clip[0].set_ylabel(r'$1 - N_{clip}/N$', labelpad=YLABELPAD)
    axes_isi[0].set_ylabel(r'$p(\tau)$ in 1/s', labelpad=YLABELPAD)
    axes_ac[0].set_ylabel('ACH in 1/s', labelpad=YLABELPAD)
    axes_cc[0].set_ylabel('CCH in 1/s', labelpad=YLABELPAD)

    plot_clipped_firing_rate_and_eff_moved(axes_clip)

    plot_statistical_analysis_of_single_rate(
        axes_isi, axes_ac, axes_cc)

    plot_firing_rate_change(axis_step)
    plot_eff_moved(axis_moved)
    plot_cv_change(axis_cv)

    for data_id, data_type in enumerate(DATA_TYPES):
        axes[0][data_id].set_title(data_type)

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

    for axes_in_row in axes:
        ylims = [axis.get_ylim() for axis in axes_in_row]
        ylim = (min(ylim[0] for ylim in ylims),
                max(ylim[1] for ylim in ylims))

        for axis in axes_in_row:
            axis.set_ylim(ylim)

    handles, labels = axes_isi[0].get_legend_handles_labels()
    pos_x, pos_y = 2, 0
    legend_loc = (
        left_move + delta_down + (1-(delta_down+delta_up))/3*pos_x
        + (1-(delta_down+delta_up))/3 + 0.003,
        delta_down + (1-(delta_down+delta_up)-2*epsilon)/4*(3-pos_y)
        + epsilons[pos_y] + 0.02)  # loc=(x, y)
    legend = fig.legend(handles, labels, fontsize=8., loc=legend_loc)
    frame = legend.get_frame()
    frame.set_facecolor('0.9')
    frame.set_edgecolor('0.9')

    fig.savefig(f'{PLOT_PATH}/{FIG_NAME}', dpi=300)


if __name__ == '__main__':
    plot_statistics_overview()
