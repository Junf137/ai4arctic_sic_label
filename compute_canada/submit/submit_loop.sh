#!/bin/bash 
set -e
array=(

# "configs/consine_restarts/dw_10_fl_32_sgd_lr_1.py"
# "configs/consine_restarts/dw_10_fl_32_sgd_lr_01.py"
"configs/consine_restarts/dw_10_fl_32_sgd_lr_001_weights_1_1_2.py"

)

wandb_project=task_weights

for i in "${!array[@]}"; do
   # bash test_echo.sh ${array[i]} ${array2[i]}
   sbatch train_infer.sh ${array[i]} $wandb_project
   # echo  ${array[i]} $wandb_project
   echo "task successfully submitted" 
   sleep 10

done
