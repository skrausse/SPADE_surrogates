import numpy as np
import quantities as pq
import neo
import elephant.spike_train_generation as stg
import elephant.statistics as stat
import matplotlib.pyplot as plt
from scipy.special import erf
import os

if not os.path.exists('figures'):
    os.makedir('figures')

np.random.seed(0)

t_start = 0.*pq.ms
t_stop = 1000.*pq.ms

shape_factor = 2.
n_spiketrains = 1000
sampling_period = 0.1 * pq.ms
homogeneous_case = True
times = np.arange(t_start.magnitude, t_stop.magnitude, sampling_period.magnitude)
equilibrium = True

if homogeneous_case:
    rate = 50.*pq.Hz
else:
    frequency = (2. * np.pi)/ t_stop.magnitude
    rate = 100.*np.sin(frequency * times - np.pi/2)**2
    rate = neo.AnalogSignal(
        signal=rate,
        units=pq.Hz,
        sampling_period=sampling_period,
        t_start=t_start,
        t_stop=t_stop)

if homogeneous_case:
    trial_list = stg.StationaryGammaProcess(
        rate=rate, shape_factor=shape_factor, t_start=t_start, t_stop=t_stop, equilibrium=equilibrium
    ).generate_n_spiketrains(n_spiketrains)
else:
    trial_list = [stg.inhomogeneous_gamma_process(rate, shape_factor, as_array=False).rescale(t_stop.units)
                 for _ in range(n_spiketrains)]

CORRECTION = False
times = np.arange(t_start.magnitude, t_stop.magnitude, sampling_period.magnitude)
# using Shinomoto rate estimation
if all(len(trial) > 10 for trial in trial_list):
    rates = []
    for trial in trial_list:
        # Go step by step through the Shimazaki rate estimation procedure

        # Calculate optimal band width
        width_sigma = stat.optimal_kernel_bandwidth(
            trial.magnitude, times=None, bootstrap=False)['optw']

        # TAKE CARE: sigma is in seconds for non-homogeneous case
        #         width_sigma = 0.015

        unit_factor = trial.units.rescale(pq.ms).magnitude

        if CORRECTION:
            correction_factor = 2 / (
                    erf((t_stop.magnitude - times) / (np.sqrt(2.) * width_sigma * unit_factor))
                    - erf((t_start.magnitude - times) / (np.sqrt(2.) * width_sigma * unit_factor)))
        #         print(width_sigma, '(s)')

        # construct Gaussian kernel
        kernel = stat.kernels.GaussianKernel(width_sigma * trial.units)

        # calculate instaneous rate, discard extra dimension
        instantenous_rate = stat.instantaneous_rate(
            spiketrain=trial,
            sampling_period=sampling_period,
            kernel=kernel)
        if CORRECTION:
            # multiply with correction factor for stationary rate
            instantenous_rate *= correction_factor[:, None]
            # ensure integral over firing rate yield the exact number of spikes
            instantenous_rate *= len(trial) / \
                                 (np.mean(instantenous_rate, axis=0).magnitude * t_stop.rescale(pq.s).magnitude)
        rates.append(instantenous_rate)
else:
    rates = [stat.instantaneous_rate(
        spiketrain=trial,
        sampling_period=sampling_period,
        kernel=stat.kernels.GaussianKernel(
            sigma=sigma)
    ) for trial in trial_list]

# plot estimated rate
fig, ax = plt.subplots()
if homogeneous_case:
    ax.plot(rates[0].times, np.repeat(rate, len(rates[0])), label='rate')
    ax.plot(rates[0].times, np.mean(rates, axis=0)[:, 0], label='rate estimation')
else:
    ax.plot(rate.times, rate, label='rate')
    ax.plot(rate.times, np.mean(rates, axis=0)[:, 0], label='rate estimation')

n_bins = 10
hist, bars = np.histogram(np.hstack(trial_list),
                          bins=n_bins, range=(t_start.item(), t_stop.item()))
ax.plot(bars[1:] - (bars[1] - bars[0]) / 2,
        hist / n_spiketrains * (n_bins / t_stop.rescale(pq.s).item()),
        label="PSTH")
ax.set_ylabel('rate Hz')
ax.set_xlabel('time in ms')
ax.set_ylim(25, 55)
ax.legend()
if CORRECTION:
    ax.set_title('Difference rate vs. estimation (corrected)')
    fig.savefig('figures/rate_estimation_corrected.png')
else:
    ax.set_title('Difference rate vs. estimation')
    fig.savefig('figures/rate_estimation.png')

# plot estimated rate
fig, ax = plt.subplots()
if homogeneous_case:
    ax.plot(rates[0].times, np.repeat(rate, len(rates[0])), label='rate')
    ax.plot(rates[0].times, rates[0], label='rate estimation')
else:
    ax.plot(rate.times, rate, label='rate')
    ax.plot(rate.times, rates[0], label='rate estimation')
for spike in trial_list[0]:
    ax.axvline(spike.item(), color='C2')
# hist, bars = np.histogram(np.hstack(trial_list),
#                           bins=10, range=(t_start.item(), t_stop.item()))
# ax.plot(bars[1:]-(bars[1]-bars[0])/2, hist/(5/100*n_spiketrains))
ax.set_ylabel('rate Hz')
ax.set_xlabel('')
ax.legend()
if CORRECTION:
    ax.set_title('Single Trial: Difference rate vs. estimation (corrected)')
else:
    ax.set_title('Single Trial: Difference rate vs. estimation')