#!/bin/bash
#SBATCH --job-name=experiment_eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=pascalnodes
#SBATCH --output=%j_out.log
#SBATCH --error=%j_err.log

ml Anaconda3
source activate
conda activate iic
ml cuda10.0/toolkit

export CUDA_VISIBLE_DEVICES=0
python -u eval.py --config experiments/config_10_sobel.json