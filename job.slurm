#!/bin/bash
#SBATCH -p gpu
#SBATCH -A rxxxxx
#SBATCH -o bioformer_fine_tune.txt
#SBATCH -e bioformer_fine_tune.err
#SBATCH --mail-user=ara@purdue.edu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=240G

srun python biofomer_FT_QLora.py
