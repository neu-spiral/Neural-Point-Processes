#!/bin/bash

for dataset in Building 
do
    for feature in AE
    do
        for d in 7 10 32
        do
            echo "Executing training: dataset $dataset, feature $feature, mode mesh, d: $d"
            python3 npp.py --dataset $dataset --feature $feature --mode mesh --num_encoder 16 32 64  --num_decoder 64 32 --n 1000 --d $d --epochs 200 --num_runs 1 --batch_size 100 --experiment_name P100_builds --deeper
        done    
    done
done
