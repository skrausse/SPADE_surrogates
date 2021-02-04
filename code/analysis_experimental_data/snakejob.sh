#!/usr/bin/env bash
#SBATCH -o /home/a.stella/projects/SPADE_analysis/r2g_data/code/slurm/slurm-%j.out
#SBATCH -e /home/a.stella/projects/SPADE_analysis/r2g_data/code/slurm/slurm-%j.err
#SBATCH --time=96:00:00
#SBATCH --job-name=SPADE_analysis
#SBATCH --mail-type=END,FAIL

source activate snakenv
snakemake --unlock

snakemake --jobs 1000\
          --cluster "sbatch -n {cluster.n} --time {cluster.time} --mail-type=FAIL"\
          --cluster-config cluster.json\
          --jobname "{jobid}.{rulename}"\
          --keep-going\
          --rerun-incomplete

