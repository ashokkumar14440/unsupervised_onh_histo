#! /bin/bash

sbatch experiments/run_10_nosobel.sh &
sbatch experiments/run_10_sobel.sh &
sbatch experiments/run_12_nosobel.sh &
sbatch experiments/run_12_sobel.sh &
