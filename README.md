# Predicting Language-Model Memorization from Tokenizer Artifacts

This is a research codebase for a simple question:

> Can we rank string-level memorization risk using only tokenizer artifacts,
> without querying the model at prediction time?

I evaluate this in two tracks:

1. **Controlled canary injection** (train ref/target models from scratch)
2. **Pretrained cross-scale evaluation** (Pythia family)

The goal is not a production tool yet; this repo is an experimental framework
for testing the hypothesis above.

## Current results snapshot

From the latest audited runs:

- **Pretrained (Pythia-1.4B vs 160M):**
  - top-5 AUROC: `trivial=0.498`, `baseline=0.428`, `counts=0.609`, `full=0.873`, `full_gbm=0.948`
  - regression (delta_bpc): `pearson=0.545`, `spearman=0.609`
- **Pretrained (Pythia-6.9B vs 1.4B):**
  - top-5 AUROC: `trivial=0.490`, `baseline=0.555`, `counts=0.652`, `full=0.855`, `full_gbm=0.884`
  - regression (delta_bpc): `pearson=0.455`, `spearman=0.502`
- **Controlled canary run (`hpc_real_01`):**
  - top-5 AUROC: `token_count=0.659`, `len_entropy=0.920`, `full_logistic=0.985`, `full_ridge=0.977`

Interpretation: in pretrained settings, trivial features are near random while
tokenizer-structure features carry most of the useful signal.

## Project structure

```
configs/              Experiment configs (canary + pretrained YAML)
data/
  raw/                Raw input text files
  corpora/            Processed corpora with canary injection
  candidates/         Candidate strings for evaluation
src/
  run_pipeline.py     10-step canary-injection pipeline
  pretrained_eval.py  Pretrained model evaluation + ablation
  extract_validate.py Extraction-based validation
  extract_features.py Tokenizer feature extraction (+ zlib baseline)
  build_corpus.py     Corpus assembly with canary injection
  train_tokenizer.py  BPE / Unigram tokenizer training
  train_lm.py         Transformer LM training from scratch
  score_bpc.py        BPC scoring (teacher-forced)
  build_labels.py     ΔBPC label construction
  train_predictor.py  Risk predictor training with feature subsets
  eval_metrics.py     Evaluation (AUROC, AUPRC, Precision@K, TPR@FPR)
  normalize.py        Tokenizer normalization extraction
  config.py           YAML config loader and validator
scripts/
  gen_nuke_data.py    Synthetic corpus + extreme canaries
  gen_real_data.py    WikiText-103 + realistic canaries
  gen_max_data.py     Large-scale diverse canary generation
  gen_pile_candidates.py  Candidates for pretrained evaluation
  run_pretrained.py   Run pretrained eval from YAML config
  hpc_pretrained.sh   SLURM script for HPC pretrained runs
  hpc_smoke.sh        HPC smoke test
  show_results.py     Display pipeline results
notebooks/
  colab_run.ipynb     Google Colab execution interface
docs/
  HPC_RUNBOOK.md      Cluster deployment guide
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick start: Canary-injection experiments

```bash
# Smoke test (~10 min on T4 GPU)
python scripts/gen_nuke_data.py
python -m src.run_pipeline --config configs/colab_nuke.yaml

# Full-scale (~3 hours on T4 GPU)
python scripts/gen_max_data.py
python -m src.run_pipeline --config configs/colab_max.yaml
```

## Quick start: Pretrained model evaluation

```bash
# Generate candidates
python scripts/gen_pile_candidates.py --n-per-bucket 1000

# Pythia-1.4B vs 160M (primary experiment, ~3h on A100)
python scripts/run_pretrained.py --config configs/pythia_1b.yaml

# With extraction validation
python scripts/run_pretrained.py --config configs/pythia_1b.yaml --extract
```

## HPC usage

```bash
# Canary-injection on HPC
sbatch scripts/hpc_smoke.sh

# Pretrained evaluation on HPC
sbatch scripts/hpc_pretrained.sh pythia_1b
sbatch scripts/hpc_pretrained.sh pythia_6b
```

## Feature groups (ablation)

The ablation framework trains predictors with increasingly rich feature sets:

| Group | Features | Purpose |
|-------|----------|---------|
| `trivial` | len_chars | Lower bound |
| `baseline` | + char_entropy, zlib_bpc | Model-free baseline |
| `counts` | + n_tokens, compression_ratio | Tokenizer counting features |
| `full` | + tok_rank_*, merge_rank_*, piece_score_* | Tokenizer artifact features |

The marginal AUROC/AUPRC gain from `counts` -> `full` is the main signal for
"tokenizer artifact value" beyond trivial string statistics.

## Reproducibility notes

- Most long runs were executed on HPC (not committed in full under `results/`).
- Small audited summaries are stored under `artifacts/audit/`.
- Some helper configs/scripts in this repo are exploratory and may not all be
  used in the final paper tables.

## Known limitations

- Single tokenizer family in pretrained track (GPT-NeoX tokenizer).
- Some tables are based on single split/seed; confidence intervals are pending.
- Top-1% and especially top-0.1% labels are sparse and noisier than top-5%.

## Specification

See [SPEC.md](SPEC.md) for the authoritative source of truth on definitions,
threat model, evaluation protocol, and smoke-test criteria.
