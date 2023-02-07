#!/bin/bash 
set -e
array=(
"configs/feature_selection/Ex_1_input_features_setup1.py"
"configs/feature_selection/Ex_1_input_features_setup2.py"
"configs/feature_selection/Ex_1_input_features_setup3.py"
"configs/feature_selection/Ex_1_input_features_setup4.py"
"configs/feature_selection/Ex_1_input_features_setup5.py"
"configs/feature_selection/Ex_1_input_features_setup6.py"
"configs/feature_selection/Ex_1_input_features_setup_all.py"
)

wandb_project=feature_variation

for i in "${!array[@]}"; do
   # bash test_echo.sh ${array[i]} ${array2[i]}
   sbatch train_infer.sh ${array[i]} $wandb_project
   # echo  ${array[i]} $wandb_project
   echo "task successfully submitted" 
   sleep 10

done

