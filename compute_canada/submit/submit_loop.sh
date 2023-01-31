#!/bin/bash 
set -e
array=(
#
"configs/setup1.py"  
# "configs/custom_configs/igarss/faster_rcnn_r50_fpn_512.py"
# "configs/custom_configs/igarss/faster_rcnn_r50_fpn_768.py"
# "configs/custom_configs/igarss/faster_rcnn_r50_fpn_1024.py"

)


for i in "${!array[@]}"; do
   # bash test_echo.sh ${array[i]} ${array2[i]}
   sbatch train_infer.sh ${array[i]}
   echo "task successfully submitted" 
   sleep 10

done
