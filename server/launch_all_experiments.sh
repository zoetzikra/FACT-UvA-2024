#!/bin/bash

# entire script fails if a single command fails
set -e

# script to be run all experiments
source launch_3var-nc-confidence-rest.sh
source launch_3var-causal-all-conf.sh
source launch_5var-skill-all-conf.sh
source launch_7var-covid-all-conf.sh