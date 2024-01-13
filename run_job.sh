#!/bin/bash

confidence_levels=(0.75 0.85 0.9 0.95)
savepath="./runs/test_0"
mkdir -p "$savepath"
dirs_to_create=("3var-nc" "3var-c" "5var-skill" "7var-covid")
for directory in "${dirs_to_create[@]}"
do
    mkdir -p "${savepath}/${directory}"
done


for confidence in "${confidence_levels[@]}"
do  
    # Run the first experiment
    python scripts/run_experiments.py 3var-noncausal 4000 200 $confidence 300 ${savepath}/3var-nc/ 3 --NGEN 600 --POP_SIZE 300 --n_digits 1 --nr_refits 5 --predict_individualized True
    
    # Run the second experiment
    python scripts/run_experiments.py 3var-causal 4000 200 $confidence 300 ${savepath}/3var-c/ 3 --NGEN 600 --POP_SIZE 300 --n_digits 1 --nr_refits 5 --predict_individualized True

    # Run the third experiment
    python scripts/run_experiments.py 5var-skill 4000 200 $confidence 300 ${savepath}/5var-skill/ 3 --NGEN 1000 --POP_SIZE 500 --n_digits 0 --nr_refits 5 --predict_individualized True --model_type rf

    # Run the fourth experiment
    python scripts/run_experiments.py 7var-covid 20000 200 $confidence 2999 ${savepath}/7var-covid/ 3 --NGEN 700 --POP_SIZE 300 --n_digits 1 --nr_refits 5 --predict_individualized True --model_type rf
done

