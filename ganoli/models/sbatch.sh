#!/bin/bash

#SBATCH --job-name=training
#SBATCH --output=training.out
#SBATCH --error=training.error
#SBATCH --mem=32000
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate ganoli
python GanoliModel.py
