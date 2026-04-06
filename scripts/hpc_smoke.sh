#!/usr/bin/env bash
# HPC smoke test: validates env + full pipeline (requires GPU node).
# Usage:
#   chmod +x scripts/hpc_smoke.sh
#   ./scripts/hpc_smoke.sh
#
# Or paste commands into an interactive GPU session.

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -d .venv ]]; then
  echo "Create a venv first: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt datasets"
  exit 1
fi
# shellcheck source=/dev/null
source .venv/bin/activate

pip install -q -r requirements.txt
pip install -q datasets

echo "=== Download corpus + candidates (WikiText + colab_real-style files) ==="
python scripts/gen_real_data.py

echo "=== Run pipeline (hpc_smoke config) ==="
python -m src.run_pipeline --config configs/hpc_smoke.yaml

echo "=== Done. Check results/hpc_smoke/hpc_smoke_pipeline.json ==="
