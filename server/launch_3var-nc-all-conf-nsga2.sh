#!/bin/bash

# entire script fails if a single command fails
set -e

# script should be run from the project directory
PROJECT_DIR="$PWD"

# project should have a src directory
SRC_DIR="$PROJECT_DIR"/server
LOGS_RESULTS_DIR="$PROJECT_DIR"/results_reduced_logs/3var-nc-logs-nsga2/
RUNS_RESULTS_DIR="$PROJECT_DIR"/results_reduced/3var-nc-collected-nsga2/

mkdir -p "$RUNS_RESULTS_DIR"
mkdir -p "$LOGS_RESULTS_DIR"

# confidence_levels=(0.75 0.85 0.90 0.95)
confidence_levels=(0.75 0.95)

for confidence in "${confidence_levels[@]}"
do  
    JOB_NAME=3var-nc-confidence-${confidence}-reduced
    CONFIDENCE=$confidence
    echo "Staring ${JOB_NAME} ..."
    sbatch --job-name "$JOB_NAME" "$SRC_DIR"/3var-nc-nsga2.job 3var-noncausal 2000 100 $CONFIDENCE 300 "$RUNS_RESULTS_DIR" 3 --NGEN 300 --POP_SIZE 150 --n_digits 1 --nr_refits 5 --predict_individualized True --parallelise --genetic_algo nsga2
done