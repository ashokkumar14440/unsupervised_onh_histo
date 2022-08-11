#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:4
#SBATCH --partition=pascalnodes-medium
#SBATCH --output=%j_out.log
#SBATCH --error=%j_err.log

ml Anaconda3
source activate
conda activate ashok7
ml cuda10.0/toolkit

export CUDA_VISIBLE_DEVICES=0,1,2,3
python -u eval.py --config config.json
