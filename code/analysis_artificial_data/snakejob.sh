#!/usr/bin/env bash

source activate surrogates
snakemake --unlock\
          --configfile configfile.yaml\
	  --use-conda\
	  --cores 1

snakemake --jobs 1000\
          --cluster "sbatch -n {cluster.n} --time {cluster.time} --mail-type=FAIL --mem={cluster.mem} --partition=blaustein -o /users/bouss/SPADE_anda/code/slurm/rule-%j.out -e /users/bouss/SPADE_anda/code/slurm/rule-%j.err"\
          --cluster-config cluster.json\
          --jobname "{jobid}.{rulename}"\
	  --latency-wait 90\
          --keep-going\
          --rerun-incomplete\
	  --cores 160