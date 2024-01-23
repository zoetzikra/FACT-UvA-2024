#!/bin/bash

# entire script fails if a single command fails
set -e

# script should be run from the project directory
PROJECT_DIR="$PWD"

# project should have a src directory
SRC_DIR="$PROJECT_DIR"/server
LOGS_RESULTS_DIR="$PROJECT_DIR"/7var-covid-logs-resume/
RUNS_RESULTS_DIR="$PROJECT_DIR"/results/7var-covid/

mkdir -p "$LOGS_RESULTS_DIR"

rm -rf "$RUNS_RESULTS_DIR"/N20000_Nrec200_gam0.9_t0.5_lbd2999.0_nit3_NGEN700_POPS300_74/2/

rm -rf "$RUNS_RESULTS_DIR"/N20000_Nrec200_gam0.75_t0.5_lbd2999.0_nit3_NGEN700_POPS300_943/2/

rm -rf "$RUNS_RESULTS_DIR"/N20000_Nrec200_gam0.85_t0.5_lbd2999.0_nit3_NGEN700_POPS300_36/2/

rm -rf "$RUNS_RESULTS_DIR"/N20000_Nrec200_gam0.95_t0.5_lbd2999.0_nit3_NGEN700_POPS300_400/2/

# confidence_levels=(0.75 0.85 0.90 0.95)
# for confidence in "${confidence_levels[@]}"
# do  
#     JOB_NAME=7var-covid-confidence-${confidence}
#     CONFIDENCE=$confidence
#     sbatch --job-name "$JOB_NAME" "$SRC_DIR"/7var-covid.job 7var-covid 20000 200 $CONFIDENCE 2999 "$SAVE_DIR" 3 --NGEN 700 --POP_SIZE 300 --n_digits 1 --nr_refits 5 --predict_individualized True --model_type rf --id 696 --parallelise --genetic_algo nsga3
# done


CONFIDENCE=0.75
JOB_NAME=7var-resume-covid-confidence-${CONFIDENCE}
sbatch --job-name "$JOB_NAME" "$SRC_DIR"/7var-covid-resume.job 7var-covid 20000 200 $CONFIDENCE 2999 "$RUNS_RESULTS_DIR" 3 --NGEN 700 --POP_SIZE 300 --n_digits 1 --nr_refits 5 --predict_individualized True --model_type rf --id 943 --parallelise 


CONFIDENCE=0.85
JOB_NAME=7var-resume-covid-confidence-${CONFIDENCE}
sbatch --job-name "$JOB_NAME" "$SRC_DIR"/7var-covid-resume.job 7var-covid 20000 200 $CONFIDENCE 2999 "$RUNS_RESULTS_DIR" 3 --NGEN 700 --POP_SIZE 300 --n_digits 1 --nr_refits 5 --predict_individualized True --model_type rf --id 36 --parallelise


CONFIDENCE=0.90
JOB_NAME=7var-resume-covid-confidence-${CONFIDENCE}
sbatch --job-name "$JOB_NAME" "$SRC_DIR"/7var-covid-resume.job 7var-covid 20000 200 $CONFIDENCE 2999 "$RUNS_RESULTS_DIR" 3 --NGEN 700 --POP_SIZE 300 --n_digits 1 --nr_refits 5 --predict_individualized True --model_type rf --id 74 --parallelise

CONFIDENCE=0.95
JOB_NAME=7var-resume-covid-confidence-${CONFIDENCE}
sbatch --job-name "$JOB_NAME" "$SRC_DIR"/7var-covid-resume.job 7var-covid 20000 200 $CONFIDENCE 2999 "$RUNS_RESULTS_DIR" 3 --NGEN 700 --POP_SIZE 300 --n_digits 1 --nr_refits 5 --predict_individualized True --model_type rf --id 400 --parallelise 