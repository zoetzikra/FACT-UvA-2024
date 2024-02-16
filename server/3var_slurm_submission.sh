#!/bin/bash

classifiers=("logreg")
datasets=("3var-noncausal" "3var-causal")
confidences=("0.75" "0.85" "0.9" "0.95")

base_dir="../final_results"

for classifier in "${classifiers[@]}"
do
    for dataset in "${datasets[@]}"
    do
    	for confidence in "${confidences[@]}"
    	do
            result_dir="${base_dir}/"
            mkdir -p "$result_dir"
            sbatch server/experiment.job ${dataset} 4000 200 ${confidence} 300 ${result_dir} 10 --NGEN 600 --POP_SIZE 300 --n_digits 1 --nr_refits 5 --predict_individualized True --model_type ${classifier} --parallelise --seed 1
	done
    done
done
