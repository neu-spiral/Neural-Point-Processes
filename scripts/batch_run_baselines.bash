#!/bin/bash
#SBATCH --job-name=run_baseline
#SBATCH --output=./sbatch_output/run_gpu.out
#SBATCH --error=./sbatch_output/run_gpu.err
#################
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1                       
#SBATCH --mem=32Gb
#SBATCH --gres=gpu:v100-sxm2:1 
#SBATCH --time=1:00:00
module load anaconda3/2022.05
module load cuda/11.0
source activate pycox

python3 npp.py --dataset $1 --feature $2 --mode $3 --n $4 --d $5 --n_pins $6 --partial_percent $7 --epochs 100 --num_runs 3
