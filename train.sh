#!/bin/bash

#SBATCH --job-name=ganoli
#SBATCH --output=logs/runs/ganoli_%j.out
#SBATCH --error=logs/runs/ganoli_%j.error
#SBATCH --mem=32000
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1 --constraint=ampere
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rogerjin@mit.edu

source ~/.bashrc
conda activate ganoli
cd /om2/user/rogerjin/GANOLI
python train.py --config=configs/longt5_encoder.json
