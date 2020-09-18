#!/bin/bash

# This script work by iteratively giving as parameter the epoch and the p_id to sbatch (40 combination of them in total)
# This params are given as a parameter to the job.sl that we've defined as the first input parameters
# Meaning that we should call this bash script like this:
# ./run_pli_generation.bash [job_name.sl]
# Where job_name.sl is the name of the slurm file that we want to iteratively batch for scheduling.

# Variable we will iterate around
EPOCHS=("eyesclosed_1" "induction" "emergence_first" "emergence_last" "eyesclosed_8")
P_IDS=("MDFA03" "MDFA05" "MDFA06" "MDFA07" "MDFA10" "MDFA11" "MDFA12" "MDFA15" "MDFA17")

# batching loop
for p_id in ${P_IDS[@]}; do
    for epoch in ${EPOCHS[@]}; do
        echo "${p_id} for state ${epoch}"
        sbatch --export=P_ID=$p_id,EPOCH=$epoch $1
    done
done
