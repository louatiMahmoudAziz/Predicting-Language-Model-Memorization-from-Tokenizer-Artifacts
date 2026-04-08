#!/usr/bin/env bash
#SBATCH -J exp1-pythia1b
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 10:00:00

set -euo pipefail

REPO=/mnt/projects/mansours_project/Predicting-Language-Model-Memorization-from-Tokenizer-Artifacts
cd "$REPO"
source .venv/bin/activate

export PYTHONPATH="$REPO:${PYTHONPATH:-}"

# <- IMPORTANT (fixes "No module named src")
mkdir -p logs

python scripts/gen_pile_candidates.py --n-per-bucket 1000
python scripts/run_pretrained.py --config configs/pythia_1b.yaml
