#!/bin/bash

# We assume running this from the script directory

EPOCHS= ("ind" "emf5" "eml5" "ec8") 
GRAPHS= ("aec" "pli" "both")
STEPS= ("01" "10")


for graph in ${GRAPHS[@]}; do
    for epoch in ${EPOCHS[@]}; do
        for feature in ${STEPS[@]}; do 
            analysis_param="${graph}_${epoch}_${steps}"
            echo "${analysis_param}"
            sbatch --export=ANALYSIS_PARAM=$analysis_param $1
        done

    done

done
