#!/bin/bash 
set -e
array=(

configs/down_scaling_variation/down_scale_by_10_time_location.py

)

wandb_project=task_weights

for i in "${!array[@]}"; do
   # bash test_echo.sh ${array[i]} ${array2[i]}
   sbatch train_infer.sh ${array[i]} $wandb_project
   # echo  ${array[i]} $wandb_project
   echo "task successfully submitted" 
   sleep 10

done
