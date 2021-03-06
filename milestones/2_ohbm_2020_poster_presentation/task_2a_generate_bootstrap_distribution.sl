#!/bin/bash
#SBATCH --job-name=bootstrap
#SBATCH --account=def-sblain
#SBATCH --mem=90000      # increase as needed
#SBATCH --time=0-3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mail-user=yacine.mahdid@mail.mcgill.ca # adjust this to match your email address
#SBATCH --mail-type=ALL

module load python/3.7.4

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index scikit-learn
pip install --no-index pandas
python generate_bootstrap_distribution.py $ANALYSIS_PARAM