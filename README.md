# Predicting Language-Model Memorization from Tokenizer Artifacts

Can tokenizer-derived features alone predict whether a string is at risk of
verbatim memorization by a language model — **without access to model weights,
logprobs, or the training corpus at prediction time?**

This repository implements two complementary experimental tracks:

1. **Controlled (canary injection):** Train matched ref/target LMs from scratch,
   inject canaries at known repetition counts, predict memorization via ΔBPC.

2. **Pretrained (Pythia family):** Evaluate on real pretrained models (70M → 6.9B)
   where the evaluator does not control the training data. Uses cross-scale
   ΔBPC, zlib baselines, and extraction validation.

## Key results

Run experiments and check `results/` for metrics. The ablation framework
measures the **marginal contribution** of tokenizer-specific features over
trivial baselines (string length, character entropy, zlib compression).

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

The marginal AUROC/AUPRC gain from `counts` → `full` measures the unique
contribution of tokenizer artifacts beyond what trivial features provide.

## Specification

See [SPEC.md](SPEC.md) for the authoritative source of truth on definitions,
threat model, evaluation protocol, and smoke-test criteria.
