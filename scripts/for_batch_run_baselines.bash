#!/bin/bash

#activate your env before running this

for dataset in PinMNIST #Synthetic Building #
do
    for feature in AE DDPM
    do
        for modality in PS-RGBNIR #PS-RGBNIR-SAR PS-RGB  
        do
            for sigma in 0 0.1 0.2 0.5 1
            do
                for d in 3 10 #32
                do
                    mode="mesh"
                    n_pins=10                   
                    name="${dataset}_${mode}"
                    echo "Executing training: dataset $dataset, feature $feature, mode mesh, d: $d"
                    sbatch ./scripts/batch_run_baselines.bash $dataset $feature $mode $d $n_pins $name $sigma $modality
                done
                for n_pins in 10 100
                do
                    mode="random"
                    d=10
                    name="${dataset}_${mode}"
                    echo "Executing training: dataset $dataset, feature $feature, mode random, npins $n_pins"
                    sbatch ./scripts/batch_run_baselines.bash $dataset $feature $mode $d $n_pins $name $sigma $modality
                done
            done
        done
    done
done