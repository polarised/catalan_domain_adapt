#!/bin/bash
#SBATCH --job-name=MLM_Train_CA
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -D /fhome/amlai09/catalan_domain_adapt
#SBATCH -t 0-00:10:00
#SBATCH -p tfgm
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH -o %x_%u_%j.out
#SBATCH -e %x_%u_%j.err

# Activate virtual environment
source /fhome/amlai09/catalan_domain_adapt/venv/bin/activate

# Check GPU status
echo "Running on node: $(hostname)"
nvidia-smi

# Run Python training script
echo "Starting Python training script..."
python3 mlm.py \
    --train-batch-size 8 \
    --eval-batch-size 8 \
    --num-train-epochs 3 \
    --use-group-texts \
    --output-dir /tmp/test_run_logs

echo "Training finished."
