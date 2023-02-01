#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gpus-per-node=v100:1 # request a GPU
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8 # change this parameter to 2,4,6,... and increase "--num_workers" accordingly to see the effect on performance
#SBATCH --mem=25G
#SBATCH --time=2:00:00
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
# get the basename for the config file, basename is an inbuilt shell command
config_basename=$(basename $config .py) 
python quickstart.py $1



echo "Starting testing"
python test_upload.py $1 $CHECKPOINT