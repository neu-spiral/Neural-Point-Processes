#!/bin/bash
# source /home/shi.cheng/anaconda3/etc/profile.d/conda.sh
# conda activate dsk
my_dir = /work/DNAL/shi.cheng/NPP/Satellite_Fusion
# cd $pwd

for dataset in PinMNIST Sythetic Building
do
    for feature in AE DDPM
    do
        for mode in mesh random
        do
            for n in '100' '1000'
            do
                for d in '3' '10'
                do
                    for n_pins in '10' '100'
                    do
                        for partial_percent in '0.2' '0.5' '0.8'
                        do
                            sbatch /my_dir/batch_run_baselines.bash $dataset $feature $mode $n $d $n_pins $partial_percent
                        do
                    do
                done
            done
        done
    done
done