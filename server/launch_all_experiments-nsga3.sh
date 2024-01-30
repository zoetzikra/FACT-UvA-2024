#!/bin/bash

# entire script fails if a single command fails
set -e

# script to be run all experiments
source launch_3var-causal-all-conf-nsga3.sh
source launch_3var-nc-all-conf-nsga3.sh
source launch_7var-covid-all-conf-nsga3.sh