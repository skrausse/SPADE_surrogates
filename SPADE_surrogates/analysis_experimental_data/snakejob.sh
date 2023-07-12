#!/bin/bash
#SBATCH --time=96:00:00   								                                # walltime
#SBATCH --mem=16G	   								                                    # memory per CPU core
#SBATCH -J "SPADE surrogates"                                                           # job name
#SBATCH --mail-user=s.krausse@fz-juelich.de   						                    # email address
#SBATCH --mail-type=END									                                # notify on job completed
#SBATCH --mail-type=FAIL								                                # notify on job failed
#SBATCH --output=../../../01_cluster_output/surrogates_%j.out                           # redirect cluster output
#SBATCH --error=../../../01_cluster_output/surrogates_%j.err                            # redirect cluster errors
#SBATCH --partition=hamstein2022                                                        # Specify which partition to use

module load mpi/openmpi

# source activate patterns

snakemake --unlock\
	  --configfile ../configfile.yaml
          --cores 1

snakemake --jobs 1000\
          --cluster-config cluster.json\
          --cluster "sbatch -n {cluster.n}\
                            --time {cluster.time}\
                            --mail-type=FAIL\
                            --mail-user=s.krausse@fz-juelich.de\
                            --mem={cluster.mem}\
	                        --partition=hamstein2022\
	                        -o ../../../01_cluster_output/surr_job_%j.out\
                            -e ../../../01_cluster_output/surr_job_%j.err"\
          --jobname "{jobid}.{rulename}"\
	      --latency-wait 90\
          --keep-going\
          --rerun-incomplete\
	      --cores 160