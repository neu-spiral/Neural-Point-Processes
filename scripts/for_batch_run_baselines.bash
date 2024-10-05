#!/bin/bash

# Activate your environment before running this

# Define kernel parameters for each kernel type
declare -A kernel_params
# 0 is MSE and the rests are RBF or SM
kernel_params["RBF"]="0 0.1 0.2 0.5 2"
kernel_params["SM"]="1 2 3 4 5"

# Define ranges for d, n_pins, kernel modes, and learning rates
d_values=("32" "10")
n_pins_values=("100") #"10" 
kernel_modes=("learned" "predicted") #"fixed") # 
learning_rates=("1e-3" "1e-4") # 

for dataset in Building #PinMNIST #Synthetic    #
do
    for feature in AE #DDPM
    do
        for mode in "random" #"mesh" 
        do
            for kernel in "SM" "RBF" 
            do
                for kernel_param in ${kernel_params[$kernel]}
                do
                    for kernel_mode in "${kernel_modes[@]}"
                    do
                        for lr in "${learning_rates[@]}"
                        do
                            if [ "$mode" == "random" ]; then
                                for n_pins in "${n_pins_values[@]}"    
                                do
                                    d=10  # Fixed n_pins for random mode
                                    name="${dataset}_${mode}_${n_pins}"
                                    echo "Executing: $dataset, $feature, $mode, d: $d, n_pins: $n_pins, kernel: $kernel, kernel_mode: $kernel_mode, param: $kernel_param, lr: $lr"
                                    sbatch ./scripts/batch_run_baselines.bash $dataset $feature $mode $d $n_pins $kernel $kernel_mode $kernel_param $lr $name
                                done
                            else  # mode is "mesh"
                                for d in "${d_values[@]}"
                                do
                                    n_pins=10  # Fixed d for mesh mode
                                    name="${dataset}_${mode}_${d}"
                                    echo "Executing: $dataset, $feature, $mode, d: $d, n_pins: $n_pins, kernel: $kernel, kernel_mode: $kernel_mode, param: $kernel_param, lr: $lr"
                                    sbatch ./scripts/batch_run_baselines.bash $dataset $feature $mode $d $n_pins $kernel $kernel_mode $kernel_param $lr $name
                                done
                            fi
                        done
                    done
                done
            done
        done
    done
done