#!/bin/bash 
set -e
array=(
"configs/feature_selection/Ex_3_input_features_HH_HV_Position.py"
"configs/feature_selection/Ex_3_input_features_HH_HV_Time_Position_linear_int.py"
"configs/feature_selection/Ex_3_input_features_HH_HV_Time_Position.py"
"configs/feature_selection/Ex_3_input_features_HH_HV_Time.py"
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

