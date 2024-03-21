#!/bin/bash

#activate your env before running this

for dataset in Building #PinMNIST Synthetic
do
    for feature in DDPM
    do
         for d in 10 32
         do
             for mode in mesh
             do
                 for n_pins in 10
                 do
                     for name in Mesh_new
                     do
                         for sigma in 0 0.1 0.2
                         do
                            echo "Executing training: dataset $dataset, feature $feature, mode mesh, d: $d"
                            sbatch /work/DNAL/shi.cheng/NPP/Satellite_Fusion/scripts/batch_run_baselines.bash $dataset $feature $mode $d $n_pins $name $sigma
                         done
                     done
                 done
             done
         done
        for n_pins in 10 100
        do
            for mode in random
            do
                for d in 10
                do
                    for name in Random_new
                    do
                        for sigma in 0 0.1 0.2
                        do
                            echo "Executing training: dataset $dataset, feature $feature, mode random, npins $n_pins"
                            sbatch /work/DNAL/shi.cheng/NPP/Satellite_Fusion/scripts/batch_run_baselines.bash $dataset $feature $mode $d $n_pins $name $sigma
                        done
                    done
                done
            done
        done
    done
done