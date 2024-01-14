#!/bin/bash

#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G 
#SBATCH --constraint=rome
#SBATCH --partition=batch
#SBATCH --mail-type=ALL
#SBATCH --output=./3var-nc-logs/%x-%j-slurm.out
#SBATCH --error=./3var-nc-logs/%x-%j-slurm.err

source $HOME/miniconda3/bin/activate ./env


export CODE_SERVER_CONFIG=~/.config/code-server/config.yaml
node=$(/bin/hostname)
port=10121
user=${USER} 
submit_host=ilogin

if [ -f ${CODE_SERVER_CONFIG} ] ; then
 rm ${CODE_SERVER_CONFIG}
fi

echo "bind-addr: ${node}:${port}" >> $CODE_SERVER_CONFIG 
echo "auth: password" >> $CODE_SERVER_CONFIG
echo "password: qwerty1234!" >> $CODE_SERVER_CONFIG
echo "cert: false" >> $CODE_SERVER_CONFIG

echo "Copy the following line in a new terminal to create a secure SSH tunnel between your computer and Ibex compute node."
echo "ssh -L localhost:${port}:${node}:${port} ${user}@${submit_host}.ibex.kaust.edu.sa"

code-server --auth=password --verbose