#!/bin/bash 
set -e
array=(
"configs/feature_selection/Ex_1_input_features_setup1.py"
"configs/feature_selection/Ex_1_input_features_setup2.py"
"configs/feature_selection/Ex_1_input_features_setup3.py"
"configs/feature_selection/Ex_1_input_features_setup4.py"
"configs/feature_selection/Ex_1_input_features_setup5.py"
"configs/feature_selection/Ex_1_input_features_setup6.py"
)


for i in "${!array[@]}"; do
   # bash test_echo.sh ${array[i]} ${array2[i]}
   sbatch train_infer.sh ${array[i]}
   echo "task successfully submitted" 
   sleep 10

done
