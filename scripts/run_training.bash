#!/bin/bash

for dataset in Synthetic PinMNIST 
do
    for feature in AE DDPM
    do
        for d in 3 10
        do
            echo "Executing training: dataset $dataset, feature $feature, mode mesh, d: $d"
            python3 npp.py --dataset $dataset --feature $feature --mode mesh --num_encoder 64 32  --num_decoder 64 --n 1000 --d $d --epochs 100 --num_runs 1
        done    
        for n_pins in 10 100
        do
            echo "Executing training: feature $feature , mode random, npins $n_pins"
            python3 npp.py --dataset $dataset --feature $feature --mode random --num_encoder 64 32  --num_decoder 64 --n 1000 --n_pins $n_pins --epochs 100 --num_runs 1
        done
    done
done

for dataset in Synthetic PinMNIST 
do
    for feature in AE DDPM
    do
        for d in 3 10
        do
            echo "Executing training: dataset $dataset, feature $feature, mode mesh, d: $d"
            python3 npp.py --dataset $dataset --feature $feature --mode mesh --num_encoder 64 32  --num_decoder 64 --n 1000 --d $d --epochs 100 --num_runs 1 --deeper
        done    
        for n_pins in 10 100
        do
            echo "Executing training: feature $feature , mode random, npins $n_pins"
            python3 npp.py --dataset $dataset --feature $feature --mode random --num_encoder 64 32  --num_decoder 64 --n 1000 --n_pins $n_pins --epochs 100 --num_runs 1 --deeper
        done
    done
done