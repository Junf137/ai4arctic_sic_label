#!/bin/bash

# Usage: bash batch_submit.sh <submit_num> <wandb_project> <seed> <venv_path>
if [ "$#" -ne 5 ]; then
    echo "---* Usage: bash ${0##*/} <job_script> <submit_num> <wandb_project> <seed> <venv_path>"
    exit 1
fi

# Set the variables
job_script=$1
submit_num=$2
wandb_project=$3
seed=$4
venv_path=$5

# Set the config array
config_array=(
    "configs/sic_label/sic_label.py"
)


# loop submission
for i in "${!config_array[@]}"; do
    echo "sbatch $job_script ${config_array[i]} $wandb_project $seed $venv_path"

    for j in $(seq 1 $submit_num); do
        sbatch $job_script ${config_array[i]} $wandb_project $seed $venv_path
        sleep 3
    done
done
