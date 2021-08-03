import numpy as np
import os
import elephant.spade as spade
import argparse
import quantities as pq

# Session and context to analyze
parser = argparse.ArgumentParser(
    description='Define parameter for filtering the results'
                ' of the SPADE analysis on R2G')

parser.add_argument('context', metavar='context', type=str,
                    help='behavioral context (epoch_trialtype) to analyze')
parser.add_argument('session', metavar='session', type=str,
                    help='Recording session to analyze')
parser.add_argument('surrogate method', metavar='surr_method', type=str,
                    help='Surrogate method to use')
args = parser.parse_args()

# Session to analyze
session_name = args.session
# Context to analyze
context = args.context
# surr_method to use
surr_method = args.surr_method

# Initialize n_surr , pv_spec and psr_param to None
n_surr = None
psr_param = None
pv_spec = None

# Merging and filtering the results from all epochs nad trialtypes
directory = '../../results/experimental_data/{}/{}/{}'.format(surr_method,
                                                              session_name,
                                                              context)
concepts = []
ns_sgnt = []
# Collecting outputs from  all the jobs
for path in [x[0] for x in os.walk(directory)][1:]:
    f = []
    print(path)
    for (dirpath, dirnames, filenames) in os.walk(path):
        if len(filenames) > 1:
            raise ValueError('More than 1 file in the folder')
        if len(filenames) == 0:
            continue
        filename = filenames[0]
        results, loading_param, spade_param, configfile_param = np.load(
            path + '/' + filename, encoding='latin1', allow_pickle=True)
        print('n concepts', len(results['patterns']))
        concepts.extend(results['patterns'])
        alpha = configfile_param['alpha']
        correction = configfile_param['correction']
        psr_param = configfile_param['psr_param']
        # SPADE parameters
        spectrum = configfile_param['spectrum']
        winlen = configfile_param['winlen']
        binsize = configfile_param['binsize'] * pq.s
        n_surr = configfile_param['n_surr']
        min_spikes = configfile_param['abs_min_spikes']
        min_occ = configfile_param['abs_min_occ']
        if n_surr > 0:
            pv_spec = results['pvalue_spectrum']
            print('len pv_spec', len(pv_spec))
            # PSF filtering
            if len(pv_spec) == 0:
                ns_sgnt.extend([])
            elif len(pv_spec) > 0 and alpha not in {None, 1}:
                # Computing non-significant entries of the spectrum applying
                # the statistical correction
                ns_sgnt.extend(spade.test_signature_significance(
                    pv_spec=pv_spec,
                    concepts=concepts,
                    alpha=alpha,
                    winlen=winlen,
                    corr=correction,
                    report='non_significant',
                    spectrum=spade_param['spectrum']))

# Filter concepts with pvalue spectrum (psf)
if n_surr not in {0, None} and len(ns_sgnt) > 0:
    concepts = [concept for concept in concepts
                if spade._pattern_spectrum_filter(concept=concept,
                                                  ns_signatures=ns_sgnt,
                                                  spectrum=spectrum,
                                                  winlen=winlen)]
    print('n patt after psf', len(concepts))
elif n_surr == 0:
    pv_spec = None
# PSR filtering
# Decide whether filter the concepts using psr
if psr_param is not None:
    # Filter using conditional tests (psr)
    concepts = spade.pattern_set_reduction(concepts=concepts,
                                           ns_signatures=ns_sgnt,
                                           winlen=winlen,
                                           spectrum=spectrum,
                                           h_subset_filtering=psr_param[0],
                                           k_superset_filtering=psr_param[1],
                                           l_covered_spikes=psr_param[2],
                                           min_spikes=min_spikes,
                                           min_occ=min_occ)

# Reformatting output
patterns = spade.concept_output_to_patterns(concepts=concepts,
                                            pv_spec=pv_spec,
                                            winlen=winlen,
                                            binsize=binsize,
                                            spectrum=spectrum)
# Attaching additional info on neurons involved in patterns
annotations = np.load(f'../results/{surr_method}/{session_name}/{context}'
                      f'/annotations.npy', allow_pickle=True).item()

for patt_idx, selected_patt in enumerate(patterns):
    selected_patt['channel_ids'] = []
    selected_patt['unit_ids'] = []
    selected_patt['connector_aligned_ids'] = []
    # Id of neurons in the list of spike trains (sts)
    neurons_ids = selected_patt['neurons']
    # One subplot per unit
    for idx, neu_id in enumerate(neurons_ids):
        annotation = annotations[neu_id]
        selected_patt['channel_ids'].append(annotation['channel_id'])
        selected_patt['unit_ids'].append(annotation['unit_id'])
        selected_patt['connector_aligned_ids'].append(
            annotation['connector_aligned_id'])

filter_param = {'alpha': alpha,
                'psr_param': psr_param,
                'correction': correction,
                'winlen': winlen,
                'binsize': binsize,
                'spectrum': spectrum,
                'n_surr': n_surr}

# Store merged filtered results
np.save(
    f'../../results/experimental_data/{surr_method}/{session_name}/{context}'
    f'/filtered_res.npy',
    [patterns, loading_param, filter_param, configfile_param, spade_param])
