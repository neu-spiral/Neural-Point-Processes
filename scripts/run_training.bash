#!/bin/bash
module unload anaconda3
module load cuda/11.0
source activate env

for dataset in Building PinMNIST Synthetic
do
    for feature in AE DDPM
    do
        for d in 10 
        do
            echo "Executing training: dataset $dataset, feature $feature, mode mesh, d: $d"
            python npp.py --dataset $dataset --feature $feature --mode mesh --num_encoder 64 32 --num_decoder 64 --n 1000 --d $d --epochs 1000 --num_runs 3 --experiment_name Building_mesh_10
        done
       for n_pins in 10 100  
       do
           echo "Executing training: dataset $dataset, feature $feature, mode random, npins $n_pins"
           python npp.py --dataset $dataset --feature $feature --mode random --num_encoder 64 32 --num_decoder 64 --n 1000 --n_pins $n_pins --epochs 1000 --num_runs 3 --experiment_name Building_random_10
       done
    done
done
