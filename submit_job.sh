#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gpus-per-node=v100l:1 # request a GPU
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8 # change this parameter to 2,4,6,... and increase "--num_workers" accordingly to see the effect on performance
#SBATCH --mem=75G
#SBATCH --time=14:00:00
#SBATCH --output=compute_canada_output/%j.out
#SBATCH --account=def-dclausi
#SBATCH --mail-user=fernandopena961226@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE

module purge
module load python/3.9.6

echo "Loading module done"

source ~/AI4Artic_env/bin/activate
       


echo "Activating virtual environment done"

#cd /project/def-dclausi/share/whale/mmwhale/
cd $HOME/projects/def-dclausi/fer96/ai4arctic_challenge

echo "starting training..."


python quickstart.py $1 
python test_upload.py $1 $CHECKPOINT

