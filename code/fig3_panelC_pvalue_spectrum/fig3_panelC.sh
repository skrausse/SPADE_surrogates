#We show here the workflow to get the p-value-spectra plot.
python max_order_statistics.py
python create_independent_spiketrains.py
mpirun python analyze_independent_spiketrains.py
python plot_pvalue_spectrum.py