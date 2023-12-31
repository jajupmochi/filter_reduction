#!/bin/bash

#SBATCH --mail-user=anthony.gillioz@inf.unibe.ch
#SBATCH --mail-type=end,fail

#SBATCH --mem-per-cpu=50G
#SBATCH --cpus-per-task=18
#SBATCH --time=3-23:59:00
#SBATCH --output=/storage/homefs/ag21k209/neo_slurms/re_clf_gnn_%A_%a.out
#SBATCH --array=1-86

param_store=./script_gnn_arguments.txt

# Get first argument
dataset_name=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $1}')
degree=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $2}')
rmv_node_attr=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $3}')

# Put your code below this line
module load Workspace_Home
module load Python/3.9.5-GCCcore-10.3.0.lua
cd $HOME/graph_library/graph_reduction/filter_reduction
source venv/bin/activate

srun python main_gnn.py --dataset $dataset_name $degree $rmv_node_attr --n-trials 10 --n-outer-cv 10 --n-inner-cv 3 --n-cores-gs 2 --n-cores-cv 9 --max-epochs 800
