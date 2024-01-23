#!/bin/bash

# entire script fails if a single command fails
set -e

# script should be run from the project directory
PROJECT_DIR="$PWD"

# project should have a src directory
SRC_DIR="$PROJECT_DIR"/server
LOGS_RESULTS_DIR="$PROJECT_DIR"/5var-skill-logs-smart/
RUNS_RESULTS_DIR="$PROJECT_DIR"/results/5var-skill/

# mkdir -p "$RUNS_RESULTS_DIR"
mkdir -p "$LOGS_RESULTS_DIR"

# rm -rf "$RUNS_RESULTS_DIR"/N4000_Nrec200_gam0.9_t0.5_lbd300.0_nit3_NGEN1000_POPS500_105/1/

# rm -rf "$RUNS_RESULTS_DIR"/N4000_Nrec200_gam0.75_t0.5_lbd300.0_nit3_NGEN1000_POPS500_454/1/

# rm -rf "$RUNS_RESULTS_DIR"/N4000_Nrec200_gam0.85_t0.5_lbd300.0_nit3_NGEN1000_POPS500_352/1/

# rm -rf "$RUNS_RESULTS_DIR"/N4000_Nrec200_gam0.95_t0.5_lbd300.0_nit3_NGEN1000_POPS500_181/1/

# CONFIDENCE=0.75
# JOB_NAME=resume-5var-skill-confidence-${CONFIDENCE}
# sbatch --job-name "$JOB_NAME" "$SRC_DIR"/5var-skill-smart.job 5var-skill 4000 200 $CONFIDENCE 300 "$RUNS_RESULTS_DIR" 3 --NGEN 1000 --POP_SIZE 500 --n_digits 0 --nr_refits 5 --predict_individualized True --model_type rf --id 454

# CONFIDENCE=0.85
# JOB_NAME=resume-5var-skill-confidence-${CONFIDENCE}
# sbatch --job-name "$JOB_NAME" "$SRC_DIR"/5var-skill-smart.job 5var-skill 4000 200 $CONFIDENCE 300 "$RUNS_RESULTS_DIR" 3 --NGEN 1000 --POP_SIZE 500 --n_digits 0 --nr_refits 5 --predict_individualized True --model_type rf --id 352

# CONFIDENCE=0.90
# JOB_NAME=resume-5var-skill-confidence-${CONFIDENCE}
# sbatch --job-name "$JOB_NAME" "$SRC_DIR"/5var-skill-smart.job 5var-skill 4000 200 $CONFIDENCE 300 "$RUNS_RESULTS_DIR" 3 --NGEN 1000 --POP_SIZE 500 --n_digits 0 --nr_refits 5 --predict_individualized True --model_type rf --id 105

CONFIDENCE=0.95
JOB_NAME=resume-5var-skill-confidence-${CONFIDENCE}
sbatch --job-name "$JOB_NAME" "$SRC_DIR"/5var-skill-smart.job 5var-skill 4000 200 $CONFIDENCE 300 "$RUNS_RESULTS_DIR" 3 --NGEN 1000 --POP_SIZE 500 --n_digits 0 --nr_refits 5 --predict_individualized True --model_type rf --id 181