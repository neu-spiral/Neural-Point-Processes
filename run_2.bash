#!/bin/bash

for dataset in Building 
do
    for feature in AE
    do    
        for n_pins in 10 100 200
        do
            echo "Executing training: feature $feature , mode random, npins $n_pins"
            python3 npp.py --dataset $dataset --feature $feature --mode random --num_encoder 16 32 64 --num_decoder 64 32 --n 1000 --n_pins $n_pins --epochs 200 --num_runs 1 --batch_size 100 --experiment_name P100_builds --deeper
        done
    done
done
