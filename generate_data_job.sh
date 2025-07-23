#!/bin/bash
#SBATCH --account=def-sandyr         # replace with your PI's account
#SBATCH --gres=gpu:1                # request 1 GPU
#SBATCH --cpus-per-task=4          # adjust based on tokenizer/model needs
#SBATCH --mem=16G                  # RAM
#SBATCH --time=03:00:00  # or even longer if needed
#SBATCH --job-name=gen-text
#SBATCH --output=logs/gen_output_%j.log
#SBATCH --mail-type=END,FAIL


module load python/3.10 cuda/12.2
source venv/bin/activate
python generate_data.py
