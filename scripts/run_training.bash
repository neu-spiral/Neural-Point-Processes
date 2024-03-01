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
module load anaconda3
module load cuda
source activate npp

for dataset in Synthetic PinMNIST 
do
    for feature in AE DDPM
    do
        if [[ "$dataset" == "Synthetic" && "$feature" == "DDPM" ]]; then
            continue
        fi
        for d in 3 10
        do
            echo "Executing training: dataset $dataset, feature $feature, mode mesh, d: $d"
            python3 npp.py --dataset $dataset --feature $feature --mode mesh --n 1000 --d $d --epochs 100 --num_runs 2
        done    
        for n_pins in 10 100
        do
            echo "Executing training: feature $feature , mode random, npins $n_pins"
            python3 npp.py --dataset $dataset --feature $feature --mode random --n 1000 --d $d --n_pins $n_pins --epochs 100 --num_runs 2
        done
    done
done