#!/bin/bash
#!/bin/bash
#SBATCH --job-name=v100
#SBATCH --output=./sbatch_output/run_gpu%j.out
#SBATCH --error=./sbatch_output/run_gpu%j.err
#################
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=30Gb
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --time=8:00:00 
source Your_virtual_env

python npp.py --dataset $1 --feature $2 --mode $3 --num_encoder 64 32 --num_decoder 64 --d $4 --n_pins $5 --epochs 500 --num_runs 1 --deeper --manual_lr --kernel $6 --kernel_mode $7 --kernel_param $8 --lr $9 --experiment_name ${10} --seed 4

python npp.py --dataset $1 --feature $2 --mode $3 --num_encoder 64 32 --num_decoder 64 --d $4 --n_pins $5 --epochs 500 --num_runs 1 --deeper --manual_lr --kernel $6 --kernel_mode $7 --kernel_param $8 --lr $9 --experiment_name ${10} --seed 3

python npp.py --dataset $1 --feature $2 --mode $3 --num_encoder 64 32 --num_decoder 64 --d $4 --n_pins $5 --epochs 500 --num_runs 1 --deeper --manual_lr --kernel $6 --kernel_mode $7 --kernel_param $8 --lr $9 --experiment_name ${10} --seed 2

python npp.py --dataset $1 --feature $2 --mode $3 --num_encoder 64 32 --num_decoder 64 --d $4 --n_pins $5 --epochs 500 --num_runs 1 --deeper --manual_lr --kernel $6 --kernel_mode $7 --kernel_param $8 --lr $9 --experiment_name ${10} --seed 1

python npp.py --dataset $1 --feature $2 --mode $3 --num_encoder 64 32 --num_decoder 64 --d $4 --n_pins $5 --epochs 500 --num_runs 1 --deeper --manual_lr --kernel $6 --kernel_mode $7 --kernel_param $8 --lr $9 --experiment_name ${10} --seed 0



