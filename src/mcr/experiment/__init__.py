from mcr.experiment.compile import compile_experiments
from mcr.experiment.run import run_experiment
import jax

key_seed = jax.random.PRNGKey(0)
seed=0

def set_seed_main(seed_local,key_d):
    global key_seed, seed
    key_seed = key_d
    seed = seed_local