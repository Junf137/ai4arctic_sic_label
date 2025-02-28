#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=/home/j46lei/projects/rrg-dclausi/j46lei/ai4arctic_challenge_clean/output/%u_%x_%j.log
#SBATCH --account=rrg-dclausi
#SBATCH --mail-user=junf137@outlook.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE

export PATH="./utils:$PATH"

# Usage: sbatch train_infer.sh <config> <wandb_project> <seed> <venv_path>
if [ "$#" -ne 4 ]; then
    _error "Usage: sbatch ${0##*/} <config> <wandb_project> <seed> <venv_path>"
    exit 1
fi

# Purge all loaded modules
_echo "Purging all loaded modules..."
module --force purge

# Load necessary modules
_echo "Loading required modules..."
module load StdEnv gcc opencv/4.10.0
module load python/3.10.13


# Activate the virtual environment
source $4/bin/activate

_echo "Running the python script..."

# Change to the repo directory
cd $HOME/projects/rrg-dclausi/$USER/ai4arctic_challenge_clean

# config=$1
# # get the basename for the config file, basename is an inbuilt shell command
# config_basename=$(basename $config .py)


export WANDB_MODE=offline
python quickstart.py $1 --wandb-project=$2 --seed=$3


# # the above python script will generate a .env at the workdir/config-name/.env
# env=./work_dir/$config_basename/.env

# echo 'Reading environment file'
# # read the .env file and save them as environment variable
# while read line; do export $line; done < $env

# echo "Starting testing"
# python test_upload.py $1 $CHECKPOINT