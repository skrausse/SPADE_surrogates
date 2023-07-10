#!/bin/bash
#SBATCH --time=24:00:00   								                                # walltime
#SBATCH --ntasks=60   									                                # number of processor cores (i.e. tasks)
#SBATCH --mem=32G	   								                                    # memory per CPU core
#SBATCH -J "SPADE surrogates"                                                           # job name
#SBATCH --mail-user=s.krausse@fz-juelich.de   						                    # email address
#SBATCH --mail-type=END									                                # notify on job completed
#SBATCH --mail-type=FAIL								                                # notify on job failed
#SBATCH --output=../../../01_cluster_output/surrogates_%j.out                           # redirect cluster output
#SBATCH --error=../../../01_cluster_output/surrogates_%j.err                            # redirect cluster errors
#SBATCH --partition=hamstein                                                            # Specify which partition to use

cd ~/projects/SPADE_surrogates/SPADE_surrogates/analysis_experimental_data/snakejob.sh
module load mpi/openmpi
module load mpi/mpich/3.2
module load mpi/mpich/3.3.2

snakemake   --cores 60 \
            --rerun-incomplete  \
            --keep-going \
            --latency-wait 90 \
            --nolock \

scontrol show jobid ${SLURM_JOBID} -dd                                                  # Job summary at exit

# --jobs                maximum number of CPU cores/jobs use in parallel
# --cluster             cluster command to submit rules as jobs
# --cluster-config      file to specify cluster wildcards
# --jobname             name of snakemake submited jobs
# --latency-wait        outwait filesystem latency after jobs are finished (s)
# --keep-going          go on with independet jobs if a job fails
# --rerun-incomplete    rerun jobs where the output is incomplete
# "SBATCH --nodes 2" would fix the required nodes