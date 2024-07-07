#!/bin/bash
#!/bin/bash
#SBATCH --job-name=A100
#SBATCH --output=./sbatch_output/run_gpu%j.out
#SBATCH --error=./sbatch_output/run_gpu%j.err
#################
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=30Gb
#SBATCH --gres=gpu:a100:1
#SBATCH --time=8:00:00
source activate npp

python npp.py --dataset $1 --feature $2 --mode $3 --num_encoder 64 32 --num_decoder 64 --n 1000 --d $4 --n_pins $5 --epochs 200 --num_runs 1 --deeper --experiment_name $6 --sigmas $7 --modality $8
