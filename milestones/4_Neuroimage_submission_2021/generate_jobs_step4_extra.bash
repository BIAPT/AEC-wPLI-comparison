#!/bin/bash

# We assume running this from the script directory
GRAPHS=("aec" "pli" "both")
STEPS=("01")

for graph in ${GRAPHS[@]}; do
	for steps in ${STEPS[@]}; do 
		analysis_param="${graph}_${steps}"
		echo "${analysis_param}"
		sbatch --export=ANALYSIS_PARAM=$analysis_param $1
	done
done
