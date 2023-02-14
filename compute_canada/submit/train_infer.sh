#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gpus-per-node=v100l:1 # request a GPU
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=12 # change this parameter to 2,4,6,... and increase "--num_workers" accordingly to see the effect on performance
#SBATCH --mem=64G
#SBATCH --time=30:00:00
#SBATCH --output=/home/fer96/projects/def-dclausi/fer96/ai4arctic_challenge/compute_canada_output/%j.out
#SBATCH --account=def-dclausi
#SBATCH --mail-user=fernandopena961226@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE

module purge
module load python/3.9.6

echo "Loading module done"

source ~/Ai4Artic4/bin/activate


echo "Activating virtual environment done"

#cd /project/def-dclausi/share/whale/mmwhale/
# cd $HOME/projects/def-dclausi/AI4arctic/$USER
cd /home/fer96/projects/def-dclausi/fer96/ai4arctic_challenge

echo "starting training..."
config=$1 
# config=$1 
# # get the basename for the config file, basename is an inbuilt shell command
# config_basename=$(basename $config .py) 

python quickstart.py $1 --wandb-project=$2

# # the above python script will generate a .env at the workdir/config-name/.env
# env=./work_dir/$config_basename/.env

# echo 'Reading environment file'
# # read the .env file and save them as environment variable
# while read line; do export $line; done < $env

# echo "Starting testing"
# python test_upload.py $1 $CHECKPOINT
