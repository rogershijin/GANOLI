#!/bin/bash

#SBATCH --job-name=tensorboard
#SBATCH --output=test/sbatch/tb.out
#SBATCH --error=test/sbatch/tb.error
#SBATCH --mem=32000
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate ganoli
tensorboard --logdir=/om2/user/rogerjin/GANOLI/ganoli/models/logs --bind_all
