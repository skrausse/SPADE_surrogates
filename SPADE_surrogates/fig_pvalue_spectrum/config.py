"""
In the module the configs are all set up as dataclasses which inherit each
other. Such that every parameter can be obtained as an attribute of a class.
"""
import os
from typing import Tuple, Union, Dict
from dataclasses import dataclass

import quantities as pq


@dataclass
class PathSetUp:
    """
    Data class for the paths.
    """
    spiketrain_path: str = '../../data/pvalue_spectrum/spiketrains/'
    pattern_path: str = '../../data/pvalue_spectrum/patterns/'
    plot_path: str = '../../data/pvalue_spectrum/plots/'
    slurm_path: str = '../../data/pvalue_spectrum/slurm/'

    def __post_init__(self):
        for path in (self.spiketrain_path, self.pattern_path, self.plot_path,
                     self.slurm_path):
            if not os.path.exists(path):
                os.makedirs(path)


@dataclass
class SpadeSetUp:
    """
    Data class containing the SPADE parameters
    """
    bin_size: pq.Quantity = 5 * pq.ms
    win_len: int = 12
    min_spikes: int = 2
    min_occ: int = 2
    max_spikes: Union[None, int] = None
    max_occ: Union[None, int] = None
    min_neu: int = 1
    approx_stab_pars: Union[None, Dict] = None
    n_surrogates: int = 5000
    dither: pq.Quantity = 25 * pq.ms
    spectrum: str = '3d#'
    alpha: float = 0.05
    stat_corr: str = 'holm'
    surr_method: str = 'joint_isi_dithering'
    psr_param: Union[Tuple[int], None] = (2, 2, 2)
    output_format: str = 'patterns'
    sizes_to_analyze: Tuple[int] = (2, 3, 4, 5)

    @property
    def surr_method_short(self):
        """
        returns a short name for each surrogate method
        Returns
        -------
        str
        """
        surr_methods = {'dither_spikes': 'UD',
                        'joint_isi_dithering': 'JISI',
                        'dither_spikes_with_refractory_period': 'UDR',
                        'ground_truth': 'GT',
                        'trial_shifting': 'TR-SHIFT'}
        return surr_methods[self.surr_method]

    @property
    def surr_kwargs(self):
        surr_kwargs: dict = {'dither_spikes': {},
                             'joint_isi_dithering': {},
                             'dither_spikes_with_refractory_period': {},
                             'ground_truth': {},
                             'trial_shifting': {'trial_length': 100.*pq.ms,
                                                'trial_separation': 0.*pq.ms}}
        return surr_kwargs[self.surr_method]


@dataclass
class TestCaseSetUp(SpadeSetUp, PathSetUp):
    """
    Data class with the set up to generate the spike trains, combined with the
    SPADE and path setup.
    """
    n_spiketrains: int = 20
    rate: pq.Quantity = 60 * pq.Hz
    dead_time: pq.Quantity = 1.6 * pq.ms
    shape_factor: float = 2.
    t_start: pq.Quantity = 0*pq.s
    t_stop: pq.Quantity = 1*pq.s
    data_type: str = 'PPD'
    n_realizations: int = 100
