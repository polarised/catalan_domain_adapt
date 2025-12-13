#!/bin/bash
#SBATCH --job-name=TrainBaseline
#SBATCH -n 1                      # Number of CPU tasks
#SBATCH -N 1                      # Number of nodes
#SBATCH -D /fhome/amlai09/catalan_domain_adapt  # Working directory
#SBATCH -t 1-00:00:00             # Max runtime: 1 day
#SBATCH -p tfgm                    # Partition/queue
#SBATCH --mem=32G                  # Memory allocation
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH -o %x_%u_%j.out            # STDOUT log
#SBATCH -e %x_%u_%j.err            # STDERR log

# Activate virtual environment
source /fhome/amlai09/catalan_domain_adapt/venv/bin/activate

# Check GPU
echo "Running on node: $(hostname)"
nvidia-smi

# Create output directories if they don't exist
mkdir -p ./Training/model_output
mkdir -p ./Training/logs

# Run your Python script
echo "Starting training script..."
python3 train_baseline.py

echo "Training finished."
