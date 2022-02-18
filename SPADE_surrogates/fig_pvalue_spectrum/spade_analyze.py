"""
Module to make a SPADE run for every pattern size singularly and then
combining the results to make the pattern set reduction.
"""
import numpy as np

import spade

import create_spike_trains as create_st

try:
    from mpi4py import MPI  # for parallelized routines
except ImportError:
    rank = 0
else:
    comm = MPI.COMM_WORLD  # create MPI communicator
    rank = comm.Get_rank()  # get rank of current MPI task


class SPADE(create_st.SpikeTrainGeneration):
    """
    class to summarize the SPADE run
    """
    def run_spade(self, spiketrains):
        """
        runs the main spade functions for every size to analyze separately and
        then combines it for the PSR run.

        Parameters
        ----------
        spiketrains : List[neo.SpikeTrain]

        Returns
        -------
        spade_output : Dict
            dictionary containing the spade results in form of the p-value
            spectrum, the non-significant signatures and the patterns
        """
        if not isinstance(spiketrains, list):
            spiketrains = list(spiketrains)
        if rank == 0:
            concepts = []
            pv_spec = []

        max_binomial_statistics = np.load(
            file=f'{self.pattern_path}max_binomial_statistics.npy',
            allow_pickle=True).item()

        for size in self.sizes_to_analyze:
            mean, std = max_binomial_statistics[
                (int(self.rate), size)]
            min_occ = int(round(mean - std))
            min_occ = max(min_occ, 1)

            spade_output = spade.spade(
                spiketrains,
                binsize=self.bin_size,
                winlen=self.win_len,
                min_spikes=size,
                min_occ=min_occ,
                max_spikes=size,
                max_occ=self.max_occ,
                min_neu=size,
                n_surr=self.n_surrogates,
                dither=self.dither,
                spectrum=self.spectrum,
                alpha=None,
                stat_corr='no',
                surr_method=self.surr_method,
                psr_param=None,
                output_format='concepts',
                ground_truth=self.create_independent_spike_trains,
                **self.surr_kwargs
            )
            if rank == 0:
                concepts.extend(spade_output['patterns'])
                pv_spec.extend(spade_output['pvalue_spectrum'])

        if rank != 0:
            return None

        if len(pv_spec) > 0:
            ns_signatures = spade.test_signature_significance(
                pv_spec, concepts, self.alpha, self.win_len,
                corr=self.stat_corr,
                report='non_significant', spectrum=self.spectrum)
        if len(ns_signatures) > 0:
            concepts = \
                [concept for concept in concepts
                 if spade._pattern_spectrum_filter(
                     concept, ns_signatures, self.spectrum, self.win_len)]
            # Decide whether to filter concepts using psr
        # Filter using conditional tests (psr)
        concepts = spade.pattern_set_reduction(
            concepts, ns_signatures, winlen=self.win_len,
            spectrum=self.spectrum,
            h_subset_filtering=self.psr_param[0],
            k_superset_filtering=self.psr_param[1],
            l_covered_spikes=self.psr_param[2],
            min_spikes=min(self.sizes_to_analyze),
            min_occ=min_occ)
        patterns = spade.concept_output_to_patterns(
            concepts, self.win_len, self.bin_size, pv_spec, self.spectrum,
            spiketrains[0].t_start)
        spade_output = {
            'pvalue_spectrum': pv_spec,
            'non_sgnf_sgnt': ns_signatures,
            'patterns': patterns}
        return spade_output
