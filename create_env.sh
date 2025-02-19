#!/bin/bash

# Usage: bash create_env_new.sh <env_name>

# Ensure an environment name is provided
if [ -z "$1" ]; then
    echo "Error: No environment name provided."
    echo "Usage: bash create_env_new.sh <env_name>"
    exit 1
fi

export PATH="./utils:$PATH"

ENV_NAME=$1
ENV_BASE_DIR=~/.venvs
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
_echo "Purging all loaded modules..."
module --force purge

# Load necessary modules
_echo "Loading required modules..."
module load StdEnv/2020 gcc/9.3.0 opencv/4.8.0
module load python/3.10.2

# Create virtual environment
_echo "Creating virtual environment: $ENV_NAME in $ENV_DIR"
virtualenv "$ENV_DIR"
if [ $? -ne 0 ]; then
    echo "Error: Failed to create virtual environment."
    exit 1
fi

# Activate the virtual environment
source "$ENV_DIR/bin/activate"

# Upgrade pip and install dependencies
_echo "Upgrading pip and installing dependencies..."
pip install --no-index --upgrade pip
pip install numpy mmcv wandb==0.16.0 h5netcdf Pillow pandas\
            tqdm scikit-learn jupyterlab ipywidgets icecream \
            matplotlib xarray seaborn cmocean\
            torch torchvision torchmetrics torch-summary segmentation_models_pytorch

if [ $? -ne 0 ]; then
    echo "Error: Failed to install required Python packages."
    exit 1
fi

_echo "Environment setup complete! Activate using: source $ENV_DIR/bin/activate"
