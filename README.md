# Reproducibilty Study of: ICR (Improved-Focus Causal Recourse)


> [Antonios Tragoudaras](https://github.com/antragoudaras/), 
[Jonas Schäfer](https://github.com/schaefjo), 
[Zoe Tzifra](), 
[Akis Lionis]()
<br>

## Overview
This repo contains the code for the reproducibility study of [Improvement-Focused Causal Recourse (ICR)
](https://arxiv.org/abs/2210.15709), as part of our work during the course FACT (Fairness, Accountability, Confidentiality and Transparency in AI) of the Master of Artificial Intelligence at the University of Amsterdam (UvA).

## Outline
- [Installation](#installation)
- [Scripts to reproduce the results](#scripts-to-reproduce-the-results)
- [Plots](#plots)

## Installation
We test our code using the following environment:
```angular2html
git clone git@github.com:zoetzikra/FACT-UvA-2024.git icr
cd icr
conda env create -f icr_environment_cpu.yml 
conda activate icr-env
```
If working on a slurm cluster environment one can use the respective job scripts in the ```server/``` directory:
```angular2html
sbatch server/install_environment.job
```

## Scripts to reproduce the results
Our results, run with following user specified confidence levels ``[confidence]`` 0.75, 0.85, 0.95 and 0.90. While ```[savepath]``` being the path were you would like to store the experiment results.

**1. Generic Scipt Form**
```shell
python scripts/run_experiments.py \
    scm_name num_observations num_of_individuals_needed_recourse [confidence] 300 [savepath]/3var-nc/ [nr_runs] --NGEN 300 --POP_SIZE 150 --n_digits 1 --nr_refits 5 --model_type logreg --predict_individualized True --genetic_algo nsga2 --parallelise --robustness 
```
  - `--parallelise`: If this flag is specified, ICR, CR and CE with specified causal knowledge(individualized(SCM)/subpoppulation(Causal Graph) run in parallel to speed up our runs). 
  - `--model_type`: Prediction Model/Classifier to be used. Available options: `[logreg, rf, adaboost, SVC, MLP]`
  - `--genetic_algo`: Select the Non-dominated Sorting Genetic algorithm istance that minimizes the constrain cost objective. Available options: `[nsga2, nsga3]`
  - `--robustness`: If this flag is specified, extra experiments are run in order to assess ICR robustness in model and data shifts.
 
**2. Example of replicating our result for all ICR, CR and CE with all specified causal knowledge (SCM) using the NSGA-II genetic algorithm for synthetic (3var-causal & 3var non causal dataset) and semi-synthetic(7var-covid) datasets and also our extension on the german credit dataset (using halved hyperparameter settings)**
```shell
python scripts/run_experiments.py 3var-noncausal 2000 100 [confidence] 300 [savepath]/3var-nc-collected-nsga2/ 3 --NGEN 300 --POP_SIZE 150 --n_digits 1 --nr_refits 5 --predict_individualized True --parallelise --genetic_algo nsga2

python scripts/run_experiments.py 3var-causal 2000 100 [confidence] 300 [savepath]/3var-causal-collected-nsga2/ 3 --NGEN 300 --POP_SIZE 150 --n_digits 1 --nr_refits 5 --predict_individualized True --parallelise --genetic_algo nsga2

python scripts/run_experiments.py 7var-covid 10000 100 [confidence] 2999 [savepath]/7var-covid-collected-nsga2/ 3 --NGEN 350 --POP_SIZE 150 --n_digits 1 --nr_refits 5 --predict_individualized True --model_type rf --parallelise --genetic_algo nsga2

python scripts/run_experiments.py 7var-credit 10000 100 $CONFIDENCE 2999 "$RUNS_RESULTS_DIR" 3 --NGEN 350 --POP_SIZE 150 --n_digits 1 --nr_refits 5 --predict_individualized True --model_type rf --parallelise --genetic_algo nsga2
```
**3. Different classifiers (MLP, SVM, Adaboost) runs + robustness for 3var non causal dataset.**
```shell
python scripts/run_experiments.py 3var-noncausal 2000 100 [confidence] 300 [savepath]/robustness_shift/ 3 --NGEN 300 --POP_SIZE 150 --n_digits 1 --nr_refits 5 --model_type MLP --predict_individualized True --parallelise --robustness --shifts "(0.0, 0.5)" "(0.5, 1.0)" "(0.5, 0.5)"

python scripts/run_experiments.py 3var-noncausal 2000 100 [confidence] 300 [savepath]/robustness_shift/ 3 --NGEN 300 --POP_SIZE 150 --n_digits 1 --nr_refits 5 --model_type SVC --predict_individualized True --parallelise --robustness

python scripts/run_experiments.py 3var-noncausal 2000 100 [confidence] 300 [savepath]/robustness_shift/ 3 --NGEN 300 --POP_SIZE 150 --n_digits 1 --nr_refits 5 --model_type adaboost --predict_individualized True --parallelise --robustness
```
- `--shifts`: Expect tuple of mean and variance for the shifts. Possible to use mulitple tuple do not forget to split them using `""`

**3. Different classifiers (MLP, SVM, Adaboost) runs for 3var causal dataset.**
```shell
python scripts/run_experiments.py 3var-causal 2000 100 [confidence] 300 [savepath]/classifier/ 3 --NGEN 300 --POP_SIZE 150 --n_digits 1 --nr_refits 5 --model_type MLP --predict_individualized True --parallelise

python scripts/run_experiments.py 3var-causal 2000 100 [confidence] 300 [savepath]/classifier/ 3 --NGEN 300 --POP_SIZE 150 --n_digits 1 --nr_refits 5 --model_type SVC --predict_individualized True --parallelise

python scripts/run_experiments.py 3var-causal 2000 100 [confidence] 300 [savepath]/classifier/ 3 --NGEN 300 --POP_SIZE 150 --n_digits 1 --nr_refits 5 --model_type adaboost --predict_individualized True --parallelise
```


A complete list of the scripts to run all our experiments(in the cluster), inlcuding extensions can be found in the server folder.

## Plots 

The experiments can be compiled and combined into a plot using 

```shell
python scripts/plots.py --savepath [savepath]
```

In the ``[savepath]`` folder, for each scm you can then find two files called

- ``resultss.csv``: summary statistics for all experiment folders in the specified result folder. mean and standard deviation for
  - `eta_mean`: specified desired acceptance rate
  - `gamma_mean`: specified desired improvement rate
  - `perc_recomm_found`: percent of recourse-seeking individuals for which a recommendation could be made
  - `eta_obs_mean`: average acceptance rate (observed)
  - `gamma_obs_mean`: average improvement rate (observed)
  - `eta_obs_individualized_mean`: average acceptance rate for the individualized post-recourse predictor (observed)
  - `eta_obs_refits_batch0_mean_mean`: average acceptance rate mean over all (model multiplicity) refits on batch 1 evaluated over batch 2. 
  - `intv-cost_mean`: averge cost of the suggested interventions
  - `[...]_std`: the respective standard deviations
- ``invs_resultss.csv``: overview of interventions performed for each of the variables as well as aggregated for causes and non-causal variables

