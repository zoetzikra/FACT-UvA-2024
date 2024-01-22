#!/bin/bash

# entire script fails if a single command fails
set -e

# script should be run from the project directory
PROJECT_DIR="$PWD"

# project should have a src directory
SRC_DIR="$PROJECT_DIR"/server
SAVE_DIR="$PROJECT_DIR"/results-nsga3/7var-covid-collected/
LOGS_RESULTS_DIR="$PROJECT_DIR"/7var-covid-logs/

mkdir -p "$SAVE_DIR"
mkdir -p "$LOGS_RESULTS_DIR"

confidence_levels=(0.75 0.85 0.90 0.95)
for confidence in "${confidence_levels[@]}"
do  
    JOB_NAME=7var-covid-confidence-${confidence}
    CONFIDENCE=$confidence
    sbatch --job-name "$JOB_NAME" "$SRC_DIR"/7var-covid.job 7var-covid 20000 200 $CONFIDENCE 2999 "$SAVE_DIR" 3 --NGEN 700 --POP_SIZE 300 --n_digits 1 --nr_refits 5 --predict_individualized True --model_type rf --parallelise --genetic_algo nsga3
done
