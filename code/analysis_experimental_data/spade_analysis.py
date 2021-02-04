import numpy as np
import quantities as pq

import sys
sys.path.insert(0, './multielectrode_grasp/code/python-neo')
import argparse
try:
    from . import rgutils
except ImportError:
    import rgutils
from mpi4py import MPI
import yaml
from yaml import Loader
from spade_utils import mkdirp, split_path
from elephant.spade import spade

parser = argparse.ArgumentParser(
    description='Define parameter for SPADE analysis on R2G')
parser.add_argument('job_id', metavar='job_id', type=int,
                   help='id of the job to perform in this call')
parser.add_argument('context', metavar='context', type=str,
                   help='behavioral context (epoch_trialtype) to analyze')
parser.add_argument('session', metavar='session', type=str,
                   help='Recording session to analyze')

args = parser.parse_args()

# Session to analyze
session = args.session
# Context to analyze
context = args.context
# id of the job to perform with this call
job_id = args.job_id
# Loading parameters specific to this job id
param_dict = np.load('./param_dict.npy', encoding='latin1',
                     allow_pickle=True).item()
# Load general parameters
with open("configfile.yaml", 'r') as stream:
    config = yaml.load(stream, Loader=Loader)


# SPADE parameters
spectrum = config['spectrum']
winlen = config['winlen']
unit = config['unit']
dither = (config['dither'] * pq.s).rescale(unit)
n_surr = config['n_surr']
binsize = (config['binsize'] * pq.s).rescale(unit)
surr_method = config['surr_method']
correction = config['correction']
# Parameter specific to the job
min_occ = param_dict[session][context][job_id]['min_occ']
min_spikes = param_dict[session][context][job_id]['min_spikes']
max_spikes = param_dict[session][context][job_id]['max_spikes']
# Parameter fixed
min_neu = min_spikes
# Fix alpha and psr_param to None since statistical correction and
# psr are done in rule filter_results
alpha = 1
psr_param = None

spade_param = {'binsize': binsize,
               'winlen': winlen,
               'min_spikes': min_spikes,
               'max_spikes': max_spikes,
               'min_neu': min_spikes,
               'min_occ': min_occ,
               'n_surr': n_surr,
               'alpha': alpha,
               'psr_param': psr_param,
               'dither': dither,
               'spectrum': spectrum,
               'surr_method': surr_method,
               'correction': correction}

# Loading parameters
epoch = param_dict[session][context][job_id]['epoch']
trialtype = param_dict[session][context][job_id]['trialtype']
firing_rate_threshold = config['firing_rate_threshold']
SNR_thresh = config['SNR_thresh']
synchsize = config['synchsize']

# Check that parameter in dict correspond to parsed argument
if context != epoch + '_' + trialtype:
    raise ValueError(
        'The job_id argument does not correspond to a job of the given context')
sep = 2 * winlen * binsize
load_param = {'session': session,
              'epoch': epoch,
              'trialtype': trialtype,
              'SNR_thresh': SNR_thresh,
              'synchsize': synchsize,
              'sep': sep,
              'firing_rate_threshold': firing_rate_threshold}

# TODO: change synchrofact removal approach in rgutils
# Loading data
sts = np.load(f'../../data/concatenated_spiketrains/{session}'
              f'/{epoch}_{trialtype}.npy',
              allow_pickle=True)
sts = list(sts)

# Generate dict annotations
annotations_dict = {}
for st_idx, st in enumerate(sts):
    annotations_dict[st_idx] = st.annotations

# pop neurons out if there is a firing rate threshold
if firing_rate_threshold is not None:
    try:
        excluded_neurons = np.load('excluded_neurons.npy',
                               allow_pickle=True).item()[session]
        for neuron in excluded_neurons:
            sts.pop(int(neuron))
    except FileNotFoundError:
        print('excluded neurons list is not yet computed: '
              'run estimate_number_occurrences script')


##### SPADE analysis ######
comm = MPI.COMM_WORLD   # create MPI communicator
rank = comm.Get_rank()  # get rank of current MPI task
size = comm.Get_size()  # get number of parallel processes
print(size, 'number vp')

spade_res = spade(spiketrains=sts,
                  binsize=binsize,
                  winlen=winlen,
                  min_occ=min_occ,
                  min_spikes=min_spikes,
                  max_spikes=max_spikes,
                  min_neu=min_neu,
                  dither=dither,
                  n_surr=n_surr,
                  alpha=alpha,
                  psr_param=psr_param,
                  surr_method=surr_method,
                  spectrum=spectrum,
                  output_format='concepts')

# Path where to store the results
res_path = '../../results/experimental_data/{}/{}_{}/{}'.format(session,
                                                                epoch,
                                                                trialtype,
                                                                job_id)

# Create path is not already existing
path_temp = './'
for folder in split_path(res_path):
    path_temp = path_temp + '/' + folder
    mkdirp(path_temp)

if rank == 0:
    # Storing spade results
    np.save(
        res_path + '/results.npy', [spade_res,
                                    load_param,
                                    spade_param,
                                    config])
    # Storing annotations param
    np.save(
        '../../results/experimental_data/{}/{}_{}'.format(
            session, epoch, trialtype) + '/annotations.npy', annotations_dict)
