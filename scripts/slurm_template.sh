#!/usr/bin/env bash
#SBATCH -J memrisk-pipeline
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
# --- Edit for your cluster ---
#SBATCH -p gpu                    # partition / queue name
#SBATCH -t 08:00:00               # walltime HH:MM:SS (increase for colab_max / hpc_grid)
#SBATCH --gres=gpu:1              # one GPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# #SBATCH -A your_project_account  # uncomment if required

set -euo pipefail

# Project root (edit if job starts elsewhere)
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Load modules — examples; replace with your site (CUDA, Python)
# module purge
# module load cuda/12.1
# module load python/3.10

# Venv
source .venv/bin/activate

export PYTHONUNBUFFERED=1

# Pick ONE:
# Smoke (fast):
#   python scripts/gen_real_data.py
#   python -m src.run_pipeline --config configs/hpc_smoke.yaml

# Full Colab-scale run (~3+ h on T4-class; less on A100):
#   python scripts/gen_max_data.py
#   python -m src.run_pipeline --config configs/colab_max.yaml

python scripts/gen_real_data.py
python -m src.run_pipeline --config configs/hpc_smoke.yaml

echo "Finished at $(date)"
