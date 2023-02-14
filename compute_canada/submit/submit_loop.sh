#!/bin/bash 
set -e
array=(
"configs/feature_selection/Ex_3_input_features_HH_HV_time.py"
"configs/feature_selection/Ex_3_input_features_HH_HV.py"
)

wandb_project=feature_variation

for i in "${!array[@]}"; do
   # bash test_echo.sh ${array[i]} ${array2[i]}
   sbatch train_infer.sh ${array[i]} $wandb_project
   # echo  ${array[i]} $wandb_project
   echo "task successfully submitted" 
   sleep 10

done

