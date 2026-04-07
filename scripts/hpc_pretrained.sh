#!/bin/bash
#SBATCH --job-name=memorization-pretrained
#SBATCH --output=logs/pretrained_%j.out
#SBATCH --error=logs/pretrained_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-type=END,FAIL

# ============================================================================
# HPC SLURM script for pretrained model memorization evaluation
#
# Usage:
#   sbatch scripts/hpc_pretrained.sh pythia_1b     # Pythia-1.4B vs 160M
#   sbatch scripts/hpc_pretrained.sh pythia_6b     # Pythia-6.9B vs 1.4B
#   sbatch scripts/hpc_pretrained.sh pythia_160m   # Pythia-160M vs 70M
#   sbatch scripts/hpc_pretrained.sh pythia_70m    # Pythia-70M (zlib baseline)
#
# For A100 nodes (6.9B model), request more memory:
#   sbatch --mem=128G --gres=gpu:a100:1 scripts/hpc_pretrained.sh pythia_6b
# ============================================================================

set -euo pipefail

# --- Config name from argument ---
CONFIG_NAME="${1:-pythia_1b}"
CONFIG_FILE="configs/${CONFIG_NAME}.yaml"

echo "============================================="
echo "  Pretrained Memorization Evaluation"
echo "  Config:   ${CONFIG_FILE}"
echo "  Job ID:   ${SLURM_JOB_ID:-local}"
echo "  Node:     $(hostname)"
echo "  GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "  Date:     $(date -Iseconds)"
echo "============================================="

# --- Setup environment ---
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

# Ensure dependencies
pip install -q datasets scipy 2>/dev/null || true

# --- Verify config exists ---
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config file not found: ${CONFIG_FILE}"
    exit 1
fi

# --- Step 1: Generate candidates (if not already present) ---
CANDIDATES="data/candidates/pile_candidates.jsonl"
if [ ! -f "${CANDIDATES}" ]; then
    echo "[$(date +%H:%M:%S)] Generating candidates..."
    python scripts/gen_pile_candidates.py --n-per-bucket 2000 --output "${CANDIDATES}"
else
    echo "[$(date +%H:%M:%S)] Candidates already exist: ${CANDIDATES}"
fi

# --- Step 2: Run pretrained evaluation ---
echo "[$(date +%H:%M:%S)] Running pretrained evaluation..."
python scripts/run_pretrained.py --config "${CONFIG_FILE}"

# --- Step 3: Run extraction validation ---
echo "[$(date +%H:%M:%S)] Running extraction validation..."
python scripts/run_pretrained.py --config "${CONFIG_FILE}" --extract

echo "============================================="
echo "  COMPLETE: $(date -Iseconds)"
echo "============================================="
