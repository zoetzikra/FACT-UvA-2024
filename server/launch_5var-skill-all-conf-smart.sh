#!/bin/bash

# entire script fails if a single command fails
set -e

# script should be run from the project directory
PROJECT_DIR="$PWD"

# project should have a src directory
SRC_DIR="$PROJECT_DIR"/server
SAVE_DIR="$PROJECT_DIR"/results-nsga3/5var-skill-collected-smart/
LOGS_RESULTS_DIR="$PROJECT_DIR"/5var-skill-logs-smart/

mkdir -p "$SAVE_DIR"
mkdir -p "$LOGS_RESULTS_DIR"

confidence_levels=(0.75 0.85 0.90 0.95)
for confidence in "${confidence_levels[@]}"
do  
    JOB_NAME=5var-skill-confidence-${confidence}
    CONFIDENCE=$confidence
    sbatch --job-name "$JOB_NAME" "$SRC_DIR"/5var-skill-cores.job 5var-skill 4000 200 $CONFIDENCE 300 "$SAVE_DIR" 3 --NGEN 1000 --POP_SIZE 500 --n_digits 0 --nr_refits 5 --predict_individualized True --model_type rf --parallelise --genetic_algo nsga3
done
