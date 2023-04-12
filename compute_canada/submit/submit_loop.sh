#!/bin/bash 
set -e
array=(
"configs/course_report/baseline.py"
# "configs/course_report/donwsample.py"
)
wandb_project=course_project

for i in "${!array[@]}"; do
   # bash test_echo.sh ${array[i]} ${array2[i]}
   sbatch train_infer.sh ${array[i]} $wandb_project
   # echo  ${array[i]} $wandb_project
   echo "task successfully submitted" 
   sleep 10

done
