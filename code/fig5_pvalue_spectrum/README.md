First, create directories as 

```mkdir ../../data/pvalue_spectrum; mkdir ../../data/pvalue_spectrum/slurm```

Second, create the independent spike trains with 

```sbatch create_independent_spiketrains.jdf```

Third, run the analysis with (check before the number of tasks per node)

```sbatch analyze_independent_spiketrains.jdf```

Finally, to get the plots do

```source activate surrogates; python plot_pvalue_spectrum.py```