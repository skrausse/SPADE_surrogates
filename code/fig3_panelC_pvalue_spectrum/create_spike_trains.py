"""
Script to create independent spike trains for the analyses with SPADE.
"""
import elephant.spike_train_generation as gen

import config as cf


class SpikeTrainGeneration(cf.TestCaseSetUp):
    """
    Class to generate list of artificial, independent spike trains
    """
    def create_independent_spike_trains(self):
        """
        creates a list of independent spiketrains.

        Returns
        -------
        spiketrains: List[neo.SpikeTrain]
        """
        spiketrains = []
        for _ in range(self.n_spiketrains):
            if self.data_type == 'PPD':
                spiketrain = gen.homogeneous_poisson_process(
                    rate=self.rate, t_start=self.t_start,
                    t_stop=self.t_stop,
                    refractory_period=self.dead_time)
            elif self.data_type == 'Gamma':
                spiketrain = gen.homogeneous_gamma_process(
                    a=self.shape_factor, b=self.shape_factor * self.rate,
                    t_start=self.t_start, t_stop=self.t_stop)
            else:
                raise ValueError('data_type must be either PPD or Gamma.')
            spiketrains.append(spiketrain)
        return spiketrains

    def create_list_of_independent_spiketrains(self):
        """
        creates list of lists of independent spike trains.
        Having the dimensions number of realization times number of
        spike trains.

        Returns
        -------
        spiketrains: List[List[neo.SpikeTrain]]
        """
        list_of_spiketrains = []
        for _ in range(self.n_realizations):
            spiketrains = self.create_independent_spike_trains()
            list_of_spiketrains.append(spiketrains)
        return list_of_spiketrains
