#!/bin/bash 
set -e
array=(

"configs/patch_variation/1024.py"  
"configs/patch_variation/768.py"
"configs/patch_variation/512.py"
"configs/patch_variation/256.py"

)

wandb_project=patch_variation

for i in "${!array[@]}"; do
   # bash test_echo.sh ${array[i]} ${array2[i]}
   sbatch train_infer.sh ${array[i]} $wandb_project
   # echo  ${array[i]} $wandb_project
   echo "task successfully submitted" 
   sleep 10

done
