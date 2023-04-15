#!/bin/bash 
set -e
array=(
# "configs/feature_selection_coop/setup0.py"
# "configs/feature_selection_coop/setup1.py"
# "configs/feature_selection_coop/setup2.py"
# "configs/feature_selection_coop/setup3.py"
# "configs/feature_selection_coop/setup4.py"
# "configs/feature_selection_coop/setup5.py"
# "configs/feature_selection_coop/setup6.py"
"configs/feature_selection_coop/setup7_loc.py"
"configs/feature_selection_coop/setup7_time.py"
"configs/feature_selection_coop/setup7_time_loc.py"
# "configs/feature_selection_coop/setup8.py"
# "configs/course_report/donwsample.py"
)
wandb_project=course_project
seed=10
# seed = $RANDOM

for i in "${!array[@]}"; do
   # bash test_echo.sh ${array[i]} ${array2[i]}
   sbatch train_infer.sh ${array[i]} $wandb_project $seed
   # echo  ${array[i]} $wandb_project
   echo "task successfully submitted" 
   sleep 10

done
