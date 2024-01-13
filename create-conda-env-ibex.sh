#!/bin/bash --login

# entire script fails if a single command fails
set -e

# create the conda environment
PROJECT_DIR="$PWD"
ENV_PREFIX="$PROJECT_DIR"/env
mamba env create --prefix $ENV_PREFIX --file "$PROJECT_DIR"/icr_environment_ibex.yml --force