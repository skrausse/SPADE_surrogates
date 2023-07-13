#!/usr/bin/env bash
#SBATCH --output=../../../01_cluster_output/surrogates_%j.out
#SBATCH --error=../../../01_cluster_output/surrogates_%j.err
#SBATCH --time=96:00:00
#SBATCH --mem=32G
#SBATCH --job-name=SPADE_analysis
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=blaustein

snakemake --unlock\
	  --configfile configfile.yaml
          --use-conda\
          --cores 1

snakemake --jobs 1000\
          --cluster-config cluster.json\
          --cluster "sbatch -n {cluster.n}\
                            --time {cluster.time}\
                            --mail-type=FAIL\
                            --mem={cluster.mem}\
	                    --partition=blaustein\
	                    -o /users/krausse/spade_comparison/cluster_output/rule-%j.out\
                        -e /users/krausse/spade_comparison/cluster_output/rule-%j.err"\
          --jobname "{jobid}.{rulename}"\
          --use-conda\
	  --latency-wait 90\
          --keep-going\
          --rerun-incomplete\
	  --cores 160

