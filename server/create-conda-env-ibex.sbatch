#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=debug
#SBATCH --constraint=intel
#SBATCH --job-name=create-icr-env-ibex
#SBATCH --mail-type=ALL
#SBATCH --output=./create-icr-env-%x-%j-slurm.out
#SBATCH --error=./create-icr-env-%x-%j-slurm.err

# create the conda environment
./create-conda-env-ibex.sh