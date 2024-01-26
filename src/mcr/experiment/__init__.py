from mcr.experiment.compile import compile_experiments
from mcr.experiment.run import run_experiment

seed_main=42

def set_seed_main(seed):
    global seed_main
    seed_main = seed