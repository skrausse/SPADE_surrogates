#!/bin/bash
#SBATCH --time=96:00:00   								                                # walltime
#SBATCH --ntasks=60   									                                # number of processor cores (i.e. tasks)
#SBATCH --mem=32G	   								                                    # memory per CPU core
#SBATCH -J SPADE surrogates"                                                            # job name
#SBATCH --mail-user=s.krausse@fz-juelich.de   						                    # email address
#SBATCH --mail-type=END									                                # notify on job completed
#SBATCH --mail-type=FAIL							                                    # notify on job failed
#SBATCH --output=../../../01_cluster_output/surrogates_%j.out                           # redirect cluster output
#SBATCH --error=../../../01_cluster_output/surrogates_%j.err                            # redirect cluster errors
#SBATCH --partition=hamstein                                                            # Specify which partition to use


snakemake   --cores 60 \
            --rerun-incomplete  \
            --keep-going \
            --latency-wait 90 \
            --nolock \

scontrol show jobid ${SLURM_JOBID} -dd        
