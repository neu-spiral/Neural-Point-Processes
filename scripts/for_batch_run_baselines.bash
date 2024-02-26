#!/bin/bash
# source /home/shi.cheng/anaconda3/etc/profile.d/conda.sh
# conda activate dsk
my_dir = /work/DNAL/shi.cheng/NPP/Satellite_Fusion
# cd $pwd
n = 1000

for dataset in PinMNIST #Synthetic Building
do
    for feature in AE DDPM
    do
        for partial_percent in '0.2' '0.5' '0.8'
        do
            for mode in mesh random
            do
                if [ "$mode" == "mesh" ]; then
                    for d in '3' '10'
                    do
                        for n_pins in '10'
                        do
                            sbatch /my_dir/batch_run_baselines.bash $dataset $feature $mode $n $d $n_pins $partial_percent
                        done
                    done
                elif [ "$mode" == "random" ]; then
                    for d in '10'
                    do
                        for n_pins in '10' '100'
                        do       
                            sbatch /my_dir/batch_run_baselines.bash $dataset $feature $mode $n $d $n_pins $partial_percent
                        done
                    done
                fi
            done
        done
    done
done