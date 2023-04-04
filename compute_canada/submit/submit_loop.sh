#!/bin/bash 
set -e
array=(
"configs/edge_consitency/binary_no_water_loss.py"
"configs/edge_consitency/binary_water_loss_1.py"
"configs/edge_consitency/binary_water_loss_2_5.py"
"configs/edge_consitency/binary_water_loss_5.py"
"configs/edge_consitency/binary_water_loss_25.py"
"configs/edge_consitency/binary_water_loss_100.py"
"configs/edge_consitency/no_binary_no_water_loss.py"
"configs/edge_consitency/no_binary_water_loss_5.py"
)

wandb_project=edge_consistency

for i in "${!array[@]}"; do
   # bash test_echo.sh ${array[i]} ${array2[i]}
   sbatch train_infer.sh ${array[i]} $wandb_project
   # echo  ${array[i]} $wandb_project
   echo "task successfully submitted" 
   sleep 10

done
