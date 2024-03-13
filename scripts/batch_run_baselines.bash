#!/bin/bash
#SBATCH --job-name=USACE
#SBATCH --output=./sbatch_output/run_gpu.out
#SBATCH --error=./sbatch_output/run_gpu.err
#################
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1                       
#SBATCH --mem=32Gb
#SBATCH --gres=gpu:v100-sxm2:1 
#SBATCH --time=8:00:00
module unload anaconda3
module load cuda/11.0
source activate pycox

python npp.py --dataset $1 --feature $2 --mode $3 --num_encoder 64 32 --num_decoder 64 --n 1000 --d $4 --n_pins $5 --epochs 1000 --num_runs 3 --experiment_name Shallow
python npp.py --dataset $1 --feature $2 --mode $3 --num_encoder 64 32 --num_decoder 64 --n 1000 --d $4 --n_pins $5 --epochs 1000 --num_runs 3 --deeper --experiment_name Shallow
