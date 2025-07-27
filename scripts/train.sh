#!/bin/bash
#SBATCH --job-name=transformer_train
#SBATCH --account=def-sandyr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=logs/train-%j.out

# Load your modules and environment
module load python/3.10
source ../venv/bin/activate  # or conda activate env_name
export TOKENIZERS_PARALLELISM=false

# Optional: navigate to your code directory
cd ../model

# Run your training script
python train.py
