"""
This module makes the run to generate independent spike train data.
"""
import numpy as np

import create_spike_trains as create_st

if __name__ == '__main__':
    setup = create_st.SpikeTrainGeneration()
    list_of_spiketrains = setup.create_list_of_independent_spiketrains()

    np.save(file=f'{setup.spiketrain_path}spiketrains_'
                 f'{int(setup.rate.magnitude)}_{setup.data_type}',
            arr=list_of_spiketrains)
