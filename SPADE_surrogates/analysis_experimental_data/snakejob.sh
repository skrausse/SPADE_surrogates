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

<<<<<<< HEAD
snakemake --unlock\
	  --configfile ../configfile.yaml\
          --cores 1\
	  --rerun-incomplete

snakemake --jobs 1000\
          --cluster-config cluster.json\
          --cluster "sbatch -n {cluster.n}\
                            --time {cluster.time}\
                            --mail-type=FAIL\
                            --mail-user=s.krausse@fz-juelich.de\
                            --mem={cluster.mem}\
	                        --partition=hamstein\
	                        -o ../../../01_cluster_output/surr_job_%j.out\
                            -e ../../../01_cluster_output/surr_job_%j.err"\
          --jobname "{jobid}.{rulename}"\
	      --latency-wait 90\
          --keep-going\
          --rerun-incomplete\
	      --cores 160
=======
scontrol show jobid ${SLURM_JOBID} -dd        
>>>>>>> 6a5261aba95d2bad6cc08416557fd584814f75ad
