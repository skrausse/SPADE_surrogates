import numpy as np
import quantities as pq
import elephant.spike_train_generation as stg
import elephant.statistics as stat
import matplotlib.pyplot as plt

np.random.seed(0)

t_start = 0.*pq.ms
t_stop = 1000.*pq.ms

shape_factor = 2.
n_spiketrains = 1000
sampling_period = 0.1 * pq.ms

rate = 50.*pq.Hz

trial_list = stg.StationaryGammaProcess(
    rate=rate, shape_factor=shape_factor, t_start=t_start, t_stop=t_stop,
    equilibrium=True
).generate_n_spiketrains(n_spiketrains)

# using Shinomoto rate estimation

fig, ax = plt.subplots(figsize=(2.5, 2.5), dpi=300)
fig.tight_layout()
fig.subplots_adjust(left=0.2, bottom=0.2)
for corr_id, CORRECTION in enumerate((False, True)):
    rates = [stat.instantaneous_rate(
            spiketrain=trial,
            sampling_period=sampling_period,
            kernel='auto',
            border_correction=CORRECTION)
            for trial in trial_list]  # calculate instaneous rate,

    if corr_id == 0:
        ax.plot(rates[0].times, np.repeat(rate, len(rates[0])),
                label='original rate')

        n_bins = 10
        hist, bars = np.histogram(
            np.hstack(trial_list),
            bins=n_bins,
            range=(t_start.item(), t_stop.item()))

        ax.plot(bars[1:] - (bars[1] - bars[0]) / 2,
                hist / n_spiketrains * (n_bins / t_stop.rescale(pq.s).item()),
                label="PSTH")

    if not CORRECTION:
        # plot estimated rate
        ax.plot(rates[0].times, np.mean(rates, axis=0)[:, 0],
                label='rate estimation')

    if CORRECTION:
        ax.plot(rates[0].times, np.mean(rates, axis=0)[:, 0],
                label='corrected rate estimation')

ax.set_ylabel('rate (Hz)', fontsize=8)
ax.set_xlabel('time (ms)', fontsize=8)
ax.set_ylim(25, 52.5)
ax.tick_params(axis='x', labelsize=8)
ax.tick_params(axis='y', labelsize=8)
ax.legend(fontsize='xx-small')

# ax.set_title('Difference rate vs. estimation')
plt.show()
fig.savefig('../plots/fig_rate_estimation_corrected.png')
fig.savefig('../plots/fig_rate_estimation_corrected.eps')

