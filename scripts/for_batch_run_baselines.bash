#!/bin/bash

#activate your env before running this

for dataset in Building #PinMNIST Synthetic
do
    for feature in AE #DDPM
    do
        for d in 7 10 32
        do
            for mode in mesh
            do
                for n_pins in 10
                do
                  echo "Executing training: dataset $dataset, feature $feature, mode mesh, d: $d"
                  sbatch /work/DNAL/shi.cheng/NPP/Satellite_Fusion/scripts/batch_run_baselines.bash $dataset $feature $mode $d $n_pins
                done
            done
        done
       for n_pins in 10 100 200
       do
          for mode in random
          do
              for d in 10
              do
                echo "Executing training: dataset $dataset, feature $feature, mode random, npins $n_pins"
                sbatch /work/DNAL/shi.cheng/NPP/Satellite_Fusion/scripts/batch_run_baselines.bash $dataset $feature $mode $d $n_pins
              done
          done

       done
    done
done