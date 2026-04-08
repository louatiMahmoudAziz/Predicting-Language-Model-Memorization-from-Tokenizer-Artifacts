#!/usr/bin/env bash
#SBATCH -J exp2-pythia6b
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH -t 24:00:00

set -euo pipefail

REPO=/mnt/projects/mansours_project/Predicting-Language-Model-Memorization-from-Tokenizer-Artifacts
cd "$REPO"
source .venv/bin/activate
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
mkdir -p logs

# Optional: regenerate candidates (safe to re-run)
python scripts/gen_pile_candidates.py --n-per-bucket 1000

# Experiment 2: Pythia-6.9B vs Pythia-1.4B
python scripts/run_pretrained.py --config configs/pythia_6b.yaml
