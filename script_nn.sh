#!/bin/bash

#SBATCH --mail-user=anthony.gillioz@inf.unibe.ch
#SBATCH --mail-type=end,fail

#SBATCH --partition=epyc2
#SBATCH --qos=job_epyc2
#SBATCH --mem-per-cpu=30G
#SBATCH --cpus-per-task=10
#SBATCH --time=4-00:00:00
#SBATCH --output=/storage/homefs/ag21k209/neo_slurms/clf_nn_%A_%a.out
#SBATCH --array=1-60

param_store=./arguments_gk_spectral_big.txt

# Get first argument
dataset_name=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $1}')
degree=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $2}')

# Put your code below this line
module load Workspace_Home
module load Python/3.9.5-GCCcore-10.3.0.lua
cd $HOME/graph_library/../
source venv/bin/activate

srun python main_nn.py --root_dataset $SCRATCH/data/ --dataset $dataset_name $degree
