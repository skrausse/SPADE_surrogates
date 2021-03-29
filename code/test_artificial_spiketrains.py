import numpy as np
import yaml
import quantities as pq
import itertools

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

# for process, session, epoch, trialtype \
#         in itertools.product(processes, sessions, epochs, trialtypes):
#     # Loading data
#     sts = np.load(f'../data/artificial_data/{process}/{session}'
#                   f'/{process}_{epoch}_{trialtype}.npy',
#                   allow_pickle=True)
#     sts = list(sts)
#
#     # Generate dict annotations
#     annotations_dict = {}
#     for st_idx, st in enumerate(sts):
#         annotations_dict[st_idx] = st.annotations
#
#     # pop neurons out if there is a firing rate threshold
#     if firing_rate_threshold is not None:
#         try:
#             excluded_neurons = np.load('excluded_neurons.npy',
#                                    allow_pickle=True).item()[session]
#             for neuron in excluded_neurons:
#                 sts.pop(int(neuron))
#         except FileNotFoundError:
#             print('excluded neurons list is not yet computed: '
#                   'run estimate_number_occurrences script')
#     print(process, session, epoch, trialtype)
#     print(len(sts))
