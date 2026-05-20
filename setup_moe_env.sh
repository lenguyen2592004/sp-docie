#!/bin/bash
set -euo pipefail

# Create a virtual environment named 'moe' with conda
conda create -n moe python=3.10 -y

# Activate the virtual environment in non-interactive shell
eval "$(conda shell.bash hook)"
conda activate moe

echo "Conda environment 'moe' created and activated."