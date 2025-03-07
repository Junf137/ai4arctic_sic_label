#!/bin/bash

# Usage: bash create_env_new.sh <env_name>
if [ "$#" -ne 2 ]; then
    echo "---* Usage: bash ${0##*/} <env_folder> <env_name>"
    exit 1
fi

ENV_NAME=$2
ENV_BASE_DIR=$1
ENV_DIR=$ENV_BASE_DIR/$ENV_NAME

# Create the base directory if it doesn't exist
mkdir -p $ENV_BASE_DIR

# Check if the environment already exists
if [ -d "$ENV_DIR" ]; then
    echo "Warning: Virtual environment '$ENV_NAME' already exists in $ENV_DIR."
    read -p "Do you want to overwrite it? (y/N): " confirm
    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
        echo "Aborting."
        exit 0
    fi
    rm -rf "$ENV_DIR"  # Remove existing environment if confirmed
fi

# Purge all loaded modules
echo "---* Purging all loaded modules..."
module --force purge

# Load necessary modules
echo "---* Loading required modules..."
module load StdEnv gcc opencv/4.10.0
module load python/3.10.13

# Create virtual environment
echo "---* Creating virtual environment: $ENV_NAME in $ENV_DIR"
virtualenv "$ENV_DIR"
if [ $? -ne 0 ]; then
    echo "Error: Failed to create virtual environment."
    exit 1
fi

# Activate the virtual environment
source "$ENV_DIR/bin/activate"

# Upgrade pip and install dependencies
echo "---* Upgrading pip and installing dependencies..."
pip install --no-index --upgrade pip
pip install --ignore-installed numpy wandb==0.16.0 Pillow pandas matplotlib
pip install mmcv h5netcdf tqdm scikit-learn jupyterlab ipywidgets icecream xarray seaborn cmocean \
            torch torchvision torchmetrics torch-summary segmentation_models_pytorch

# Check whether all the packages are installed
echo "---* Checking installed packages..."
packages=("numpy" "wandb" "mmcv" "h5netcdf" "Pillow" "pandas" "tqdm" "scikit-learn" "jupyterlab" "ipywidgets" "icecream" "matplotlib" "xarray" "seaborn" "cmocean" "torch" "torchvision" "torchmetrics" "torch-summary" "segmentation_models_pytorch")
for package in "${packages[@]}"; do
    if ! python -m pip show -q "$package"; then
        echo "---* Error: Package '$package' not properly installed"
    fi
done

echo "---* Environment setup complete! Activate using: source $ENV_DIR/bin/activate"
