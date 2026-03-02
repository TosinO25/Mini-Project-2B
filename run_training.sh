#!/bin/bash
#SBATCH --job-name=emotion_large_nn
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=2
#SBATCH --time=06:00:00
#SBATCH --output=emotion_training_%j.out
#SBATCH --error=emotion_training_%j.err

# Load necessary modules
module load anaconda3

# Activate conda environment
source activate emotion_big

# Set dataset path (CHANGE THIS TO YOUR PATH)
export DATASET_ROOT="/path/to/emotion_dataset"

# Run training script
echo "Starting emotion model training on FABRIC"
echo "Dataset: $DATASET_ROOT"
date

python train_emotion_models.py

echo "Training complete"
date
