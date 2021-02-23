#!/bin/bash -l
#SBATCH --job-name=generate-wpli-graph
#SBATCH --account=def-sblain # adjust this to match the accounting group you are using to submit jobs
#SBATCH --time=0-3:00:00        # adjust this to match the walltime of your job (D-HH:MM:SS)
#SBATCH --nodes=1     
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40      # adjust this if you are using parallel commands
#SBATCH --mem=90000         # adjust this according to the memory requirement per node you need (this is MegaByte)
#SBATCH --mail-user=q2h3s6p4k0e9o7a5@biaptlab.slack.com
#SBATCH --mail-type=ALL

# Choose a version of MATLAB by loading a module:
module load matlab/2018a

# Create temporary job info location
mkdir -p /scratch/$USER/$SLURM_JOB_ID

# will run on at most 80 cores
srun matlab -nodisplay -r "generate_wpli" 

# Cleanup
rm -rf /scratch/$USER/$SLURM_JOB_ID
