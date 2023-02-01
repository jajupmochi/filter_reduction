#!/bin/bash

#SBATCH --mail-user=anthony.gillioz@inf.unibe.ch
#SBATCH --mail-type=end,fail

#SBATCH --mem-per-cpu=50G
#SBATCH --cpus-per-task=10
#SBATCH --time=3-23:59:00
#SBATCH --output=/storage/homefs/ag21k209/neo_slurms/clf_ged_%A_%a.out
#SBATCH --array=1-122

param_store=./script_ged_arguments.txt

# Get first argument
dataset_name=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $1}')
degree=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $2}')
rmv_node_attr=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $3}')


if [ "$dataset_name" = "REDDIT-MULTI-5K" ] || [ "$dataset_name" = "REDDIT-MULTI-12K" ]
then
   ALPHA="--alphas 0.5"
else
   ALPHA=""
fi

# Put your code below this line
module load Workspace_Home
module load Python/3.9.5-GCCcore-10.3.0.lua
cd $HOME/graph_library/graph_reduction/filter_reduction
source venv/bin/activate

srun python main_ged.py --dataset $dataset_name $degree $rmv_node_attr --n-trials 10 --n-outer-cv 10 --n-inner-cv 5 --n-cores 10 $ALPHA
