#!/bin/bash
#SBATCH --job-name=12_sobel
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=24G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:4
#SBATCH --partition=pascalnodes-medium
#SBATCH --output=%j_out.log
#SBATCH --error=%j_err.log

ml Anaconda3
source activate
conda activate iic
ml cuda10.0/toolkit

export CUDA_VISIBLE_DEVICES=0,1,2,3
python -u train.py --config experiments/config_12_sobel.json
