"""
Basic test if the exclusion of too high rate neurons work.
"""
import numpy as np
import yaml
import quantities as pq

with open("configfile.yaml", 'r') as stream:
    config = yaml.load(stream, Loader=yaml.Loader)

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

excluded_neurons_artificial = np.load(
    'analysis_artificial_data/excluded_neurons.npy', allow_pickle=True).item()
excluded_neurons_experimental = np.load(
    'analysis_artificial_data/excluded_neurons.npy', allow_pickle=True).item()

print(excluded_neurons_artificial)
print(excluded_neurons_experimental)
