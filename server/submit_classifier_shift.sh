#!/bin/bash

classifiers=("rf") #"rf" "SVC" "MLP" "adaboost"
datasets=("5var-skills" "7var-covid") # "3var-noncausal"
confidences=("0.75" "0.85" "0.9" "0.95")

for classifier in "${classifiers[@]}"
do
    for dataset in "${datasets[@]}"
    do
    	for confidence in "${confidences[@]}"
    	do
        	sbatch experiment.job ${dataset} 1000 100 ${confidence} 300 ../debugging/ 3 --NGEN 300 --POP_SIZE 150 --n_digits 1 --nr_refits 5 --predict_individualized True --model_type ${classifier} --robustness
	done
    done
done

