"""
Script to create Fig 6 which explains the segmentation of the experimental
data.
"""
import quantities as pq
import neo
from neo.utils import add_epoch, cut_segment_by_epoch, get_events

import matplotlib.pyplot as plt
from viziphant.rasterplot import rasterplot
from viziphant.events import add_event

import rgutils


trialtypes = 'PGHF'

# Define trigger, t_pre, t_post depending on session_name
trigger, t_pre, t_post = 'TS-ON', -350 * pq.ms, 4000 * pq.ms

# Load session, and create block depending on the trigger
session = rgutils._session('i140703-001')

block = session.read_block(
    nsx_to_load=None,
    n_starts=None,
    n_stops=None,
    channels=list(range(1, 97)),
    units='all',
    load_events=True,
    load_waveforms=False,
    scaling='raw')

data_segment = block.segments[0]
start_events = get_events(
    data_segment,
    performance_in_trial_str='correct_trial',
    trial_event_labels=trigger)
start_event = start_events[0]
epoch = add_epoch(
    data_segment,
    event1=start_event, event2=None,
    pre=t_pre, post=t_post,
    attach_result=False)
cut_trial_block = neo.Block(name="Cut_Trials")
cut_trial_block.segments = cut_segment_by_epoch(
    data_segment, epoch, reset_time=True)
selected_trial_segments = cut_trial_block.filter(
    targdict={'belongs_to_trialtype': trialtypes}, objects=neo.Segment)

seg_id = 0
seg = selected_trial_segments[seg_id]
data = []
for st in seg.filter({'sua': True}):
    # Check the SNR
    if st.annotations['SNR'] > 2.5:
        rgutils.check_snr(st, seg, seg_id)
        data.append(st.rescale(pq.ms) - t_pre)

# List of event labels that we want to consider
event_name_to_plot = ['TS-ON', 'WS-ON', 'CUE-ON', 'CUE-OFF',
                      'GO-ON', 'SR', 'RW-ON']
# Get the list of events of the first trial
all_events = seg.events[0]
# Get the most relevant events in trial
ev_idx = [i for i, val
          in enumerate(all_events.array_annotations['trial_event_labels'])
          if val in set(event_name_to_plot)]
events = all_events[ev_idx].rescale(pq.ms)

fig, axes = plt.subplots(1, 1, figsize=(5.5, 2.8), dpi=300)
fig.subplots_adjust(top=0.8, hspace=0.6, bottom=0.2)
SMALL_SIZE = 8
plt.rc('font', size=SMALL_SIZE)
markersize = 0.02

# #add labels to events
events.labels = event_name_to_plot
events.times.rescale(pq.ms)

# epochs
# start
axes.axvspan(events[0] - 250 * pq.ms, events[0] + 250 * pq.ms,
             alpha=1, color='red', fill=False,
             linewidth=3, ymin=0.01, ymax=0.98)
# cue1
axes.axvspan(events[1] - 0.25 * pq.s, events[1] + 0.25 * pq.s,
             alpha=1, color='orange', fill=False,
             linewidth=3, ymin=0.01, ymax=0.98)
# early delay
axes.axvspan(events[2] - 0 * pq.s, events[2] + 0.5 * pq.s,
             alpha=1, color='yellow', fill=False,
             linewidth=3, ymin=0.01, ymax=0.98)
# late delay
axes.axvspan(events[4] - 0.5 * pq.s, events[4] + 0. * pq.s,
             alpha=1, color='green', fill=False,
             linewidth=3, ymin=0.01, ymax=0.98)
# movement
axes.axvspan(events[5] - 0.2 * pq.s, events[5] + 0.3 * pq.s,
             alpha=1, color='blue', fill=False,
             linewidth=3, ymin=0.01, ymax=0.98)
# purple
axes.axvspan(events[6] - 0.5 * pq.s, events[6] + 0.0 * pq.s,
             alpha=1, color='purple', fill=False,
             linewidth=3, ymin=0.01, ymax=0.98)

# plots
rasterplot(data, axes=axes, s=markersize, color='k')
add_event(axes=axes, event=events, key=None)
axes.set_ylabel('Neurons', fontsize=8)
axes.set_xlabel('Time (ms)', fontsize=8)
axes.tick_params(axis="x", labelsize=8)
axes.tick_params(axis="y", labelsize=8)
plt.savefig('./plots/r2g_trial.eps', dpi=300)
plt.savefig('./plots/r2g_trial.svg', dpi=300)
