#!/bin/bash
#SBATCH --job-name=music-rec-rl
#SBATCH --output=logs/slurm_%j.log
#SBATCH --error=logs/slurm_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# ==========================================
# HPC Submission Script for Music RL Project
# ==========================================

# 1. Load Modules (Adjust based on your cluster, e.g., 'module load anaconda3')
# module load anaconda3/2023.03
# module load cuda/11.7
ml GCCcore/11.3.0 Python CUDA
# 2. Activate Environment
# Assuming 'venv' is in the current directory or provide absolute path
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "Warning: venv not found at ./venv. Please verify path."
    # fallback or exit
fi

# Print Setup Info
echo "Running on host: $(hostname)"
echo "Date: $(date)"
echo "Conda Env: $CONDA_DEFAULT_ENV"
echo "Python: $(which python)"
nvidia-smi

# 3. Run the Overnight Pipeline
# explicit python call just to be safe
python run_overnight.py

echo "Job finished at: $(date)"
