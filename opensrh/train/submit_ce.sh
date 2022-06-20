#!/bin/bash

#SBATCH --job-name=ce
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=2-00:00:00

#SBATCH --account=precisionhealth_owned1
#SBATCH --partition=precisionhealth
#SBATCH --ntasks-per-node=10
#SBATCH --mem-per-cpu=17200m

python train_ce.py -c config/train_ce.yaml 
