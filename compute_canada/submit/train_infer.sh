#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100_2g.10gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --time=5:59:00
#SBATCH --output=/home/j46lei/projects/rrg-dclausi/j46lei/ai4arctic/output/%x_%j.log
#SBATCH --account=rrg-dclausi
#SBATCH --mail-user=junf137@outlook.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE

# Usage: sbatch train_infer.sh <config> <wandb_project> <seed> <venv_path>
if [ "$#" -ne 4 ]; then
    echo "---* Usage: sbatch ${0##*/} <config> <wandb_project> <seed> <venv_path>"
    exit 1
fi

# Purge all loaded modules
echo "---* Purging all loaded modules..."
module --force purge

# Load necessary modules
echo "---* Loading required modules..."
module load StdEnv gcc opencv/4.10.0
module load python/3.10.13

# Activate the virtual environment
source $4/bin/activate

echo "---* Running the python script..."
export WANDB_MODE=offline
python quickstart.py $1 --wandb-project=$2 --seed=$3
