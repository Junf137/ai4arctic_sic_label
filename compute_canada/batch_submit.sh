#!/bin/bash

# Usage: bash batch_submit.sh <job_script> <submit_num> <venv_path> <wandb_project> <seed> <other_args...>
if [ "$#" -ne 5 ]; then
    echo "---* Usage: bash ${0##*/} <job_script> <submit_num> <venv_path> <wandb_project> <seed> <other_args...>"
    exit 1
fi

# Set the variables
job_script=$1
shift
submit_num=$1
shift
venv_path=$1
shift
wandb_project=$1
shift
seed=$1
shift
other_args="$@"

# Set the config array
config_array=(
    "configs/sic_label/masktrainval_0.0.py"
    "configs/sic_label/masktrainval_0.5.py"
    "configs/sic_label/masktrainval_1.py"
    "configs/sic_label/masktrainval_3.py"
    "configs/sic_label/masktrainval_5.py"
    "configs/sic_label/masktrainval_8.py"
    "configs/sic_label/masktrainval_35.py"
    "configs/sic_label/masktrainval_10.py"
    "configs/sic_label/masktrainval_20.py"
    "configs/sic_label/masktrainval_30.py"
    "configs/sic_label/masktrainval_40.py"
    "configs/sic_label/masktrainval_50.py"
    "configs/sic_label/masktrainval_60.py"
    "configs/sic_label/masktrainval_70.py"
    "configs/sic_label/masktrainval_80.py"
    "configs/sic_label/masktrainval_90.py"
    "configs/sic_label/masktrainval_100.py"
)


# loop submission
for i in "${!config_array[@]}"; do
    echo "sbatch $job_script $venv_path ${config_array[i]} $wandb_project $seed $other_args"

    for j in $(seq 1 $submit_num); do
        sbatch $job_script $venv_path ${config_array[i]} $wandb_project $seed $other_args
        sleep 3
    done
done
