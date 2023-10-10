#!/usr/bin/sh
#
#SBATCH --job-name="run_compression_intel"
#SBATCH --partition=compute
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=0
#SBATCH --mem-per-cpu=4G
#SBATCH --account=YOUR_ACCOUNT

module load 2022r2
module load intel/oneapi-all
module load miniconda3/4.12.0
module load cuda/11.7

ENV_NAME="venv"
ENV_FILE="environment.yml"

conda env create -n $ENV_NAME -f $ENV_FILE
conda activate $ENV_NAME

export I_MPI_PMI_LIBRARY=/cm/shared/apps/slurm/current/lib64/libpmi2.so

srun python compression_pruning.py > pi.log --pty bash

conda deactivate


