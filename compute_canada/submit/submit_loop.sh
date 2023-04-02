#!/bin/bash 
set -e
array=(
"configs/edge_consitency/binary_no_water_loss.py"
"configs/edge_consitency/binary_water_loss.py"
"configs/edge_consitency/no_binary_no_water_loss.py"
"configs/edge_consitency/no_binary_water_loss.py"
)

wandb_project=edge_consistency

for i in "${!array[@]}"; do
   # bash test_echo.sh ${array[i]} ${array2[i]}
   sbatch train_infer.sh ${array[i]} $wandb_project
   # echo  ${array[i]} $wandb_project
   echo "task successfully submitted" 
   sleep 10

done
