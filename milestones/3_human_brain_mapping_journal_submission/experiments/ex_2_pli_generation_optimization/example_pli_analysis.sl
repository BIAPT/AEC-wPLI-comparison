#!/bin/bash -l
#SBATCH --job-name=example-pli-analysis
#SBATCH --account=def-sblain # adjust this to match the accounting group you are using to submit jobs
#SBATCH --time=0-0:10:00        # adjust this to match the walltime of your job (D-HH:MM:SS)
#SBATCH --nodes=1     
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20      # adjust this if you are using parallel commands
#SBATCH --mem=90000         # adjust this according to the memory requirement per node you need (this is MegaByte)
#SBATCH --mail-user=yacine.mahdid@mail.mcgill.ca 
#SBATCH --mail-type=ALL

# Choose a version of MATLAB by loading a module:
module load matlab/2018a

# Create temporary job info location
mkdir -p /scratch/$USER/$SLURM_JOB_ID

# will run on at most 40 cores
P_ID="MDFA03"
EPOCH="eyesclosed_1"
matlab -nodisplay -r "example_pli_analysis"

# Cleanup
rm -rf /scratch/$USER/$SLURM_JOB_ID
