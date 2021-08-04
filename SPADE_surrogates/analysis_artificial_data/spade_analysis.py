import numpy as np
import quantities as pq
import argparse
from elephant.spade import spade
from mpi4py import MPI
import yaml
from yaml import Loader
from SPADE_surrogates.analyse_data_utils.spade_utils import mkdirp, split_path

parser = argparse.ArgumentParser(
    description='Define parameter for SPADE analysis on R2G')
parser.add_argument('job_id', metavar='job_id', type=int,
                    help='id of the job to perform in this call')
parser.add_argument('context', metavar='context', type=str,
                    help='behavioral context (epoch_trialtype) to analyze')
parser.add_argument('session', metavar='session', type=str,
                    help='Recording session to analyze')
parser.add_argument('process', metavar='process', type=str,
                    help='Point process to analyze')
parser.add_argument('surrogate method', metavar='surr_method', type=str,
                    help='Surrogate method to use')

args = parser.parse_args()

# Session to analyze
session = args.session
# Context to analyze
context = args.context
# id of the job to perform with this call
job_id = args.job_id
# Point process generated as artificial data
process = args.process
# surr_method to use
surr_method = args.surr_method
# Loading parameters specific to this job id
param_dict = np.load('./param_dict.npy',
                     encoding='latin1',
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

# Parameter specific to the job
min_occ = param_dict[session][process][context][job_id]['min_occ']
min_spikes = param_dict[session][process][context][job_id]['min_spikes']
max_spikes = param_dict[session][process][context][job_id]['max_spikes']

# Parameter fixed
min_neu = min_spikes
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
               'surr_method': surr_method}

# Loading parameters
SNR_thresh = config['SNR_thresh']
synchsize = config['synchsize']
epoch = param_dict[session][process][context][job_id]['epoch']
trialtype = param_dict[session][process][context][job_id]['trialtype']
firing_rate_threshold = config['firing_rate_threshold']
# Check that parameter in dict correspond to parsed argument
if context != epoch + '_' + trialtype:
    raise ValueError(
        'The job_id argument does not correspond'
        ' to a job of the given context')
sep = 2 * winlen * binsize
load_param = {'session': session,
              'epoch': epoch,
              'process': process,
              'trialtype': trialtype,
              'SNR_thresh': SNR_thresh,
              'sep': sep,
              'firing_rate_threshold': firing_rate_threshold}

# Loading data
sts = np.load(f'../../data/artificial_data/{process}/{session}'
              f'/{process}_{epoch}_{trialtype}.npy',
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
              'run Snakefile script')

# SPADE analysis
comm = MPI.COMM_WORLD  # create MPI communicator
rank = comm.Get_rank()  # get rank of current MPI task
size = comm.Get_size()  # get number of parallel processes
print(size, 'number vp')

print('min_occ', min_occ, 'min_spikes', min_spikes, 'max_spikes', max_spikes)
if surr_method == 'trial_shifting':
    spade_res = spade(
        sts, binsize, winlen, min_occ=min_occ, min_spikes=min_spikes,
        max_spikes=max_spikes, min_neu=min_neu,
        dither=dither, n_surr=n_surr, alpha=alpha,
        psr_param=psr_param, surr_method=surr_method,
        spectrum=spectrum, output_format='concepts', trial_length=500*pq.ms,
        trial_separation=sep)
else:
    spade_res = spade(
        sts, binsize, winlen, min_occ=min_occ, min_spikes=min_spikes,
        max_spikes=max_spikes, min_neu=min_neu,
        dither=dither, n_surr=n_surr, alpha=alpha,
        psr_param=psr_param, surr_method=surr_method,
        spectrum=spectrum, output_format='concepts')

# Path where to store the results
res_path = f'../../results/artificial_data/' \
    f'{surr_method}/{process}/{session}/{epoch}_{trialtype}/{job_id}'
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
        f'../../results/artificial_data/{surr_method}/{process}/'
        f'{session}/{epoch}_{trialtype}/annotations.npy',
        annotations_dict)
