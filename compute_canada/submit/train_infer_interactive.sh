#!/bin/sh
set -e 
module purge
module load python/3.9.6

echo "Loading module done"

source ~/env7_ai4arctic/bin/activate


echo "Activating virtual environment done"

#cd /project/def-dclausi/share/whale/mmwhale/
cd $HOME/projects/def-dclausi/AI4arctic/$USER

echo "starting training..."

export WANDB_MODE=offline
config=$1

config_basename=$(basename $config .py)
# echo $(pwd)
env=./work_dir/$config_basename/.env
# echo $env
# echo $(ls -a)
echo $1
echo $CHECKPOINT
python quickstart.py $1

# echo 'Reading environment file'
# while read line; do export $line; done < $env

# echo $CHECKPOINT
# python test_upload.py $1 $CHECKPOINT

