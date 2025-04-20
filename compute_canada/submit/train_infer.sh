#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100_1g.5gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=9:59:00
#SBATCH --output=/home/j46lei/projects/rrg-dclausi/j46lei/ai4arctic/output/%x_%j.log
#SBATCH --account=rrg-dclausi
#SBATCH --mail-user=junf137@outlook.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE

MIN_ARGS=4

# Usage: sbatch train_infer.sh <venv_path> <config> <wandb_project> <seed> <other_args...>
if [ $# -lt $MIN_ARGS ]; then
    echo "Error: This script requires no less than $MIN_ARGS arguments."
    echo "---* Usage: sbatch ${0##*/} <venv_path> <config> <wandb_project> <seed> <other_args...>"
    exit 1
fi

# virtual environment path
VENV_PATH=$1
shift

# config path
CONFIG_PATH=$1
shift

# wandb project name
WANDB_PROJ=$1
shift

# seed for random number generation
SEED=$1
shift

# rest of the arguments will be passed to the python script
OTHER_ARGS="$@"

# Load necessary modules
echo "---* Loading required modules..."
module --force purge
module load StdEnv gcc opencv/4.10.0 python/3.10.13
# module --force purge && module load StdEnv gcc opencv/4.10.0 python/3.10.13 && source ~/.venvs/ai4arctic/bin/activate

# Activate the virtual environment
source $VENV_PATH/bin/activate

echo "---* Running the python script..."
export WANDB_MODE=offline
# Fix the incorrect argument passing
python quickstart.py $CONFIG_PATH --wandb-project=$WANDB_PROJ --seed=$SEED $OTHER_ARGS
