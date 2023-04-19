#!/bin/bash 
set -e
array=(
# "configs/augmentation/all_ds10.py"
# "configs/augmentation/cutmix_ds10.py"
# "configs/augmentation/flip_rotation_ds10.py"
# "configs/augmentation/scale_ds10.py"
# "configs/feature_selection_coop/setup4.py"
# "configs/feature_selection_coop/setup5.py"
# "configs/feature_selection_coop/setup6.py"
# "configs/feature_selection_coop/setup7_loc.py"
# "configs/feature_selection_coop/setup7_time.py"
# "configs/feature_selection_coop/setup7_time_loc.py"
# "configs/feature_selection_coop/setup8.py"
# "configs/course_report/donwsample.py"
# "configs/loss_functions_coop/dice_loss_log.py"
# "configs/loss_functions_coop/MSE_all.py"
# "configs/loss_functions_coop/focal_loss_1.py"
# "configs/loss_functions_coop/focal_loss_2.py"
# "configs/loss_functions_coop/focal_loss_5.py"
# "configs/loss_functions_coop/focal_loss_10.py"
# "configs/loss_functions_coop/ordered_cross_entropy.`py"

# "configs/loss_functions_coop/cross_entropy_loss_task_weight.py"
# "configs/loss_functions_coop/SIC_MSE_task_weight.py"
# "configs/loss_functions_coop/cross_entropy_loss.py"
# "configs/loss_functions_coop/tversky.py"
# "configs/loss_functions_coop/tversky_log.py"
"configs/loss_functions_coop/tversky_log_0.1_0.9.py"
"configs/loss_functions_coop/tversky_log_0.2_0.8.py"
"configs/loss_functions_coop/tversky_log_0.3_0.7.py"
"configs/loss_functions_coop/tversky_log_0.4_0.6.py"
)
wandb_project=loss_coop
seed=10
# seed = $RANDOM

for i in "${!array[@]}"; do
   # bash test_echo.sh ${array[i]} ${array2[i]}
   sbatch train_infer.sh ${array[i]} $wandb_project $seed
   # echo  ${array[i]} $wandb_project
   echo "task successfully submitted" 
   sleep 10

done
