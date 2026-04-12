#!/bin/bash
#SBATCH --job-name=tokenizer_mem
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
# --- GPU (adjust partition / account to match Clipper) ---
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
# #SBATCH --account=YOUR_ACCOUNT   # uncomment if your site requires it

# Fail fast, trace optional: set -euo pipefail
set -euo pipefail

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"

# Project root = directory containing tokenizer_memorization_experiment.py
# Set this to where you cloned the repo on the cluster.
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${PROJECT_ROOT}"
echo "PROJECT_ROOT=${PROJECT_ROOT}"

mkdir -p logs

# --- Python: use module OR conda (uncomment what your site provides) ---
# module purge
# module load python/3.11
# source ~/venvs/memorization/bin/activate

if [ -f "${HOME}/venvs/memorization/bin/activate" ]; then
  # shellcheck source=/dev/null
  source "${HOME}/venvs/memorization/bin/activate"
fi

export PYTHONUNBUFFERED=1
# Hugging Face cache on scratch (recommended; data is large)
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}"

# Optional: pin visible GPU (Slurm usually sets CUDA_VISIBLE_DEVICES)
nvidia-smi || true

pip install -q -r requirements_v2.txt

# --- Run (tune --n-samples and --target-models for your thesis) ---
python tokenizer_memorization_experiment.py \
  --n-samples 100000 \
  --target-models 1.4B \
  --thresholds 5.0 1.0 \
  --n-folds 5 \
  --n-repeats 3 \
  --output-dir "results_v2_${SLURM_JOB_ID}" \
  --cache-dir "${HF_DATASETS_CACHE}" \
  ${EXTRA_ARGS:-}

echo "End: $(date)"
