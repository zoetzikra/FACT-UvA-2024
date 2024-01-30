#!/bin/bash

# entire script fails if a single command fails
set -e

# script should be run from the project directory
PROJECT_DIR="$PWD"

# project should have a src directory
SRC_DIR="$PROJECT_DIR"/server
LOGS_RESULTS_DIR="$PROJECT_DIR"/results_reduced_logs/5var-skill-logs-nsga2-logs/
RUNS_RESULTS_DIR="$PROJECT_DIR"/results_reduced/5var-skill-collected-nsga2/

mkdir -p "$RUNS_RESULTS_DIR"
mkdir -p "$LOGS_RESULTS_DIR"


confidence_levels=(0.75 0.95)
for confidence in "${confidence_levels[@]}"
do  
    JOB_NAME=5var-skill-confidence-${confidence}-reduced
    CONFIDENCE=$confidence
    echo "Staring ${JOB_NAME} ..."
    sbatch --job-name "$JOB_NAME" "$SRC_DIR"/5var-skill-nsga2.job 5var-skill 2000 100 $CONFIDENCE 300 "$RUNS_RESULTS_DIR" 3 --NGEN 5000 --POP_SIZE 250 --n_digits 0 --nr_refits 5 --predict_individualized True --model_type rf --parallelise --genetic_algo nsga2
done
