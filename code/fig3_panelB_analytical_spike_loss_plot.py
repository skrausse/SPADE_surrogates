"""
Module to plot Fig. 4 which shows the analytical spike count reduction
for PPD and Gamma spike trains and for their uniform-dithered surrogates.
"""

import math
import os

import numpy as np
import matplotlib.pyplot as plt
import quantities as pq

from scipy.special import gamma, gammainc


def firing_rate_clipped_gamma(firing_rate, shape_factor, bin_size):
    """
    Calculates the clipped firing rate for gamma spike trains

    Parameters
    ----------
    firing_rate : np.ndarray or float
    shape_factor : float
    bin_size : float

    Returns
    -------
    np.ndarray or float
    """
    if isinstance(firing_rate, np.ndarray):
        firing_rate_clipped = np.zeros(len(firing_rate))
        for i, rate in enumerate(firing_rate):
            firing_rate_clipped[i] = firing_rate_clipped_gamma(
                rate, shape_factor, bin_size)
        return firing_rate_clipped

    return firing_rate * \
       (1. - 1. / gamma(shape_factor) *
        (gamma(shape_factor) *
         gammainc(shape_factor, shape_factor * firing_rate * bin_size)
         - gamma(shape_factor + 1) *
         gammainc(shape_factor + 1,
                  shape_factor * firing_rate * bin_size) /
         (shape_factor * firing_rate * bin_size)))


def firing_rate_clipped_ppd(firing_rate, dead_time, bin_size):
    """
    Calculates the clipped firing rate for PPD spike trains

    Parameters
    ----------
    firing_rate : np.ndarray or scalar
    dead_time : float
    bin_size : float

    Returns
    -------
    np.ndarray or float
    """
    if isinstance(firing_rate, np.ndarray):
        firing_rate_clipped = np.zeros(len(firing_rate))
        for i, rate in enumerate(firing_rate):
            firing_rate_clipped[i] = firing_rate_clipped_ppd(
                rate, dead_time, bin_size)
        return firing_rate_clipped

    if bin_size > dead_time:
        lam_eff = firing_rate / (1 - firing_rate * dead_time)
        return firing_rate * dead_time / bin_size * \
            np.exp(-lam_eff * (bin_size - dead_time)) + 1 / bin_size * (
                   1. - np.exp(-lam_eff * (bin_size - dead_time)))
    return firing_rate


def firing_rate_clipped_ppd_surrogates(firing_rate, dead_time,
                                       dither_parameter, bin_size):
    """
    Calculates the clipped firing rate for uniform-dithered surrogates of
    PPD spike trains

    Parameters
    ----------
    firing_rate : pq.Quantity
    dead_time : pq.Quantity
    dither_parameter : pq.Quantity
    bin_size : pq.Quantity

    Returns
    -------
    pq.Quantity
    """
    effective_firing_rates = firing_rate / \
        (1 - (dead_time * firing_rate).simplified.magnitude)
    lambda_u = np.zeros(firing_rate.shape)
    for i in range(1, math.floor(2 * dither_parameter / dead_time) + 1):
        lambda_u += \
            (2 * dither_parameter - i * dead_time).simplified.magnitude \
            * gammainc(
                i, (effective_firing_rates * (
                    2 * dither_parameter - i * dead_time)).simplified.magnitude
            ) - (i / effective_firing_rates).simplified.magnitude * gammainc(
                i + 1, (effective_firing_rates * (
                    2 * dither_parameter - i * dead_time)).simplified.magnitude
            )
    lambda_u *= 1. / (2 * dither_parameter.simplified.magnitude ** 2)

    return firing_rate * \
        (1. - 0.5 * lambda_u * bin_size.simplified.magnitude +
         1. / 6. * (lambda_u * bin_size.simplified.magnitude) ** 2)


def firing_rate_clipped_gamma_surrogates(
        firing_rate,
        shape_factor,
        dither_parameter,
        bin_size,
        n_stop=50):
    """
    Calculates the clipped firing rate for uniform-dithered surrogates of
    gamma spike trains

    Parameters
    ----------
    firing_rate : pq.Quantity
    shape_factor : float
    dither_parameter : pq.Quantity
    bin_size : pq.Quantity
    n_stop : int, optional
        Default : 50

    Returns
    -------

    """
    lambda_u = np.zeros(firing_rate.shape)
    for i in range(1, n_stop + 1):
        lambda_u += (2 * dither_parameter).simplified.magnitude \
            * gammainc(
            i * shape_factor,
            (shape_factor * firing_rate * (2 * dither_parameter)
             ).simplified.magnitude
        ) - (i / firing_rate).simplified.magnitude * gammainc(
            i * shape_factor + 1,
            (shape_factor * firing_rate * (2 * dither_parameter)
             ).simplified.magnitude
        )
    lambda_u *= 1. / (2 * dither_parameter.simplified.magnitude ** 2)

    return firing_rate * \
        (1. - 0.5 * lambda_u * bin_size.simplified.magnitude +
         1. / 6. * (lambda_u * bin_size.simplified.magnitude) ** 2)


def plot_spike_count_reduction(
        firing_rates, bin_size, dither_parameter, dead_times, shape_factors,
        plot_path):
    """
    Plot the analytical spike count reduction
    for PPD and Gamma spike trains and for their uniform-dithered surrogates.

    Parameters
    ----------
    firing_rates : pq.Quantity
    bin_size : pq.Quantity
    dither_parameter : pq.Quantity
    dead_times : pq.Quantity
    shape_factors : np.ndarray
    plot_path : string

    Returns
    -------
    None
    """
    colors = ('C0', 'C1', 'C2', 'C3')

    fig, axes = plt.subplots(
        1, 2, figsize=(3.5, 2.5), sharey='all')
    fig.subplots_adjust(left=0.17, bottom=0.17, wspace=0.1, right=0.98)

    for dead_time_id, dead_time in enumerate(dead_times):
        spike_count_ratio = firing_rate_clipped_ppd(
            firing_rates.magnitude,
            dead_time.rescale(pq.s).magnitude,
            bin_size.rescale(pq.s).magnitude) / firing_rates.magnitude
        spike_count_reduction = 1. - spike_count_ratio

        axes[0].plot(
            firing_rates,
            spike_count_reduction,
            label=rf'd = {str(dead_time)}',
            color=colors[dead_time_id])

        spike_count_ratio_surrogates = firing_rate_clipped_ppd_surrogates(
            firing_rates,
            dead_time,
            dither_parameter,
            bin_size) / firing_rates

        spike_count_reduction_surrogates = 1. - spike_count_ratio_surrogates

        axes[0].plot(
            firing_rates,
            spike_count_reduction_surrogates,
            linestyle='--',
            color=colors[dead_time_id])

    for shape_id, shape_factor in enumerate(shape_factors):
        spike_count_ratio = firing_rate_clipped_gamma(
            firing_rates.magnitude, shape_factor,
            bin_size.rescale(pq.s).magnitude) / firing_rates.magnitude
        spike_count_reduction = 1. - spike_count_ratio

        axes[1].plot(
            firing_rates,
            spike_count_reduction,
            label=fr'$\gamma$ = {shape_factor:.1f}',
            color=colors[shape_id])

        spike_count_ratio_surrogates = firing_rate_clipped_gamma_surrogates(
            firing_rates, shape_factor,
            dither_parameter, bin_size) / firing_rates
        spike_count_reduction_surrogates = 1. - spike_count_ratio_surrogates
        axes[1].plot(
            firing_rates,
            spike_count_reduction_surrogates,
            linestyle='--',
            color=colors[shape_id])

    axes[0].set_ylabel(r'$1 - N_{clip}$/$N$', labelpad=1.5,
                       fontsize='small')
    for ax, data_type in zip(axes, ('PPD', 'Gamma')):
        ax.set_xlabel(r'$\lambda$ (Hz)', labelpad=1.5,
                      fontsize='small')
        ax.tick_params(axis='x', labelsize='small')
        ax.tick_params(axis='y', labelsize='small')
        ax.legend(fontsize='x-small')
        ax.set_title(data_type)

    fig.savefig(plot_path, dpi=300)
    plt.show()


if __name__ == '__main__':
    if not os.path.exists('../plots'):
        os.mkdir('../plots')

    # Define parameters
    shapefactors = np.arange(1., 3., 0.5)
    firingrates = np.arange(0.01, 101., 1.) * pq.Hz
    binsize = 5. * pq.ms
    deadtimes = np.arange(1.5, 3.1, 0.5) * pq.ms
    ditherparameter = 25 * pq.ms
    plotpath = '../plots/fig3_panelB_analytical_spike_count_reduction.svg'

    plot_spike_count_reduction(
        firing_rates=firingrates,
        bin_size=binsize,
        dither_parameter=ditherparameter,
        dead_times=deadtimes,
        shape_factors=shapefactors,
        plot_path=plotpath
    )
