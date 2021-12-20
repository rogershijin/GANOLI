#!/bin/bash

#SBATCH --job-name=jupyterlab
#SBATCH --output=logs/jupyterlab.out
#SBATCH --error=logs/jupyterlab.error
#SBATCH --mem=32000
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate jupyter
jupyter lab --ip=0.0.0.0 --port=1129 --no-browser --token=5521a1d0b33f2691b12462ce1fa2b858776e2cb24046cd0a
