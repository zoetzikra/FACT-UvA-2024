#!/bin/bash

# entire script fails if a single command fails
set -e

PROJECT_DIR="$PWD"

mkdir -p "$PROJECT_DIR"/results-nsga3/

# script to be run all experiments
source "$PROJECT_DIR"/server/launch_3var-nc-all-conf.sh
source "$PROJECT_DIR"/server/launch_3var-causal-all-conf.sh
source "$PROJECT_DIR"/server/launch_5var-skill-all-conf.sh
source "$PROJECT_DIR"/server/launch_7var-covid-all-conf.sh