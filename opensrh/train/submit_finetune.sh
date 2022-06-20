#!/bin/bash

#SBATCH --job-name=meow
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=2-00:00:00

#SBATCH --account=precisionhealth_owned1
#SBATCH --partition=precisionhealth
#SBATCH --ntasks-per-node=10
#SBATCH --mem-per-cpu=17200m

python train_finetune.py -c config/train_finetune.yaml
