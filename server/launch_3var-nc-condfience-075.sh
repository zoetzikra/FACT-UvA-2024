#!/bin/bash

# entire script fails if a single command fails
set -e

# script should be run from the project directory
PROJECT_DIR="$PWD"

# project should have a src directory
SRC_DIR="$PROJECT_DIR"
LOGS_RESULTS_DIR="$PROJECT_DIR"/3var-nc-logs
RUNS_RESULTS_DIR="$PROJECT_DIR"/3var-nc-runs

mkdir -p "$RUNS_RESULTS_DIR"
mkdir -p "$LOGS_RESULTS_DIR"

JOB_NAME=3var-nc-confidence-075
CONFIDENCE=0.75

sbatch --job-name "$JOB_NAME" "$SRC_DIR"/3var-nc.sbatch 3var-noncausal 4000 200 $CONFIDENCE 300 "$RUNS_RESULTS_DIR" 3 --NGEN 600 --POP_SIZE 300 --n_digits 1 --nr_refits 5 --predict_individualized True