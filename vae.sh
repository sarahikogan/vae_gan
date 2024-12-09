#!/bin/bash
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --mem=32G
#SBATCH -J "Autoencoder"
#SBATCH -p short
#SBATCH -t 12:00:00
#SBATCH --output=vae.out

module load miniconda3             # Load Python module
source ~/miniconda3/bin/activate
conda activate myenv
python_version=$(python --version)
echo "python version: $python_version"
pytorch_version=$(python -c "import torch; print(torch.__version__)")
echo "pytorch version: $pytorch_version"
echo "running python script in virtual environment!"

# Run autoencoder script
python autoencoder.py

conda deactivate