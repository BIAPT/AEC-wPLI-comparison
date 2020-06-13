#!/bin/bash

# We assume running this from the script directory
EPOCHS=("emf5" "eml5")
GRAPHS=("aec" "pli")
FEATURES=("func" "wei" "bin" "func-wei" "func-bin")


for graph in ${GRAPHS[@]}; do
    for epoch in ${EPOCHS[@]}; do
        for feature in ${FEATURES[@]}; do 
            analysis_param="${graph}_${epoch}_${feature}"
            echo "${analysis_param}"
        done

    done

    sbatch --export=ANALYSIS_PARAM=$analysis_param task_1_find_best_clf_template.sl

done
