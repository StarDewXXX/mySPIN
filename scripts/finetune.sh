#!/bin/bash -l
#SBATCH --job-name=spin
#SBATCH --time=00:10:00
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=compute #interactive
#SBATCH --output=work_dirs/slurm/%x/%j/out
#SBATCH --error=work_dirs/slurm/%x/%j/error

OUTPUT='--output slurm_outputs/%x/%j/step_%s/node_%n-rank_%t.output --error slurm_outputs/%x/%j/step_%s/node_%n-rank_%t.error '

conda activate haotian
# Set which GPU devices to be visible to the process, --num_processes should be adjusted accordingly
export CUDA_VISIBLE_DEVICES="0" #"0,1,2,3"
ACCELERATE_LOG_LEVEL=info
accelerate launch --config_file configs/deepspeed_zero3.yaml --num_processes=1 --main_process_port 2950 spin/run_spin.py configs/config.yaml --num_train_epochs=3 --output_dir="outputs/iter0-ckpt"
