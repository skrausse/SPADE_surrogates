#!/bin/bash
#SBATCH --time=96:00:00   								                                # walltime
#SBATCH --ntasks=20   									                                # number of processor cores (i.e. tasks)
#SBATCH --mem=84G	   								                                    # memory per CPU core
#SBATCH --job-name "SPADE surrogates"                                                            # job name
#SBATCH --mail-user=s.krausse@fz-juelich.de   						                    # email address
#SBATCH --mail-type=END									                                # notify on job completed
#SBATCH --mail-type=FAIL							                                    # notify on job failed
#SBATCH --output=../../../01_cluster_output/surrogates_%j.out                           # redirect cluster output
#SBATCH --error=../../../01_cluster_output/surrogates_%j.err                            # redirect cluster errors
#SBATCH --partition=hamstein                                                            # Specify which partition to use


snakemake --jobs 1000 \
          --cluster "sbatch --ntasks 1 \
                            --time 12:00:00 \
                            --mail-type=FAIL \
                            --mail-user=s.krausse@fz-juelich.de \
                            --mem=32G \
	                        --partition=hamstein \
	                        -o ../../../01_cluster_output/surr_job_%j.out \
                            -e ../../../01_cluster_output/surr_job_%j.err" \
          --jobname "{jobid}.{rulename}" \
	      --latency-wait 90 \
          --keep-going \
          --rerun-incomplete \
          --nolock \
	      --cores 160

scontrol show jobid ${SLURM_JOBID} -dd        
