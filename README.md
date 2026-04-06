# Tokenizer-Only Prediction of String-Level Memorization Risk

Research codebase for studying whether tokenizer-derived features alone can predict
whether a string is at risk of verbatim memorization by a language model.

## Project Structure

```
project/
  configs/            Experiment configuration files (YAML/JSON)
  data/
    raw/              Raw input text files
    corpora/          Processed, normalized corpora
    candidates/       Candidate strings for memorization analysis
  tokenizers/         Trained tokenizer artifacts
  models/             Trained LM and predictor artifacts
  features/           Extracted tokenizer feature matrices
  labels/             Ground-truth memorization risk labels
  results/
    metrics/          Evaluation metric summaries
    plots/            Evaluation and analysis plots
  src/                Core Python modules (see below)
  scripts/            CLI entry-point scripts
  notebooks/          Exploratory Jupyter notebooks
```

## Modules (`src/`)

| Module | Responsibility |
|---|---|
| `normalize.py` | Text normalization (Unicode, whitespace, casing) |
| `build_corpus.py` | Assemble and clean corpus from raw data |
| `train_tokenizer.py` | Train BPE/Unigram/WordPiece tokenizer on corpus |
| `extract_features.py` | Extract token-level features for candidate strings |
| `train_lm.py` | Train a language model for BPC scoring |
| `score_bpc.py` | Score candidates with bits-per-character (BPC) |
| `build_labels.py` | Construct ground-truth memorization risk labels |
| `train_predictor.py` | Train a risk predictor from features and labels |
| `eval_metrics.py` | Evaluate predictor (AUC, F1, calibration, etc.) |
| `extract_validate.py` | Validate predictions via extraction experiments |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

See `scripts/` for pipeline entry points and `notebooks/` for exploratory analysis.

### HPC (cluster)

1. Clone repo, create venv, `pip install -r requirements.txt` and `pip install datasets`.
2. Follow **`docs/HPC_RUNBOOK.md`**: smoke test with `configs/hpc_smoke.yaml` after `python scripts/gen_real_data.py`.
3. Full runs: `configs/colab_max.yaml` or prepare data for `configs/hpc_grid.yaml`.
4. Copy `scripts/slurm_template.sh`, edit `#SBATCH` lines, then `sbatch scripts/slurm_template.sh`.
