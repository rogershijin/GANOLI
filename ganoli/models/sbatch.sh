#!/bin/bash

#SBATCH --job-name=training
#SBATCH --output=test/sbatch/linear.out
#SBATCH --error=test/sbatch/linear.error
#SBATCH --mem=32000
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate ganoli
python GanoliModel.py
