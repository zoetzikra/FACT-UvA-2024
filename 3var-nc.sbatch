#!/bin/bash --login
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G 
#SBATCH --constraint=rome
#SBATCH --partition=batch
#SBATCH --mail-type=ALL
#SBATCH --output=./3var-nc-logs/%x-%j-slurm.out
#SBATCH --error=./3var-nc-logs/%x-%j-slurm.err

set -e

conda activate ./env
python scripts/run_experiments.py "$@"