# HPC runbook — what to do now

Follow this order on your cluster. Adjust partition names, modules, and paths to your site.

## 1. One-time setup (login node)

```bash
git clone https://github.com/louatiMahmoudAziz/Predicting-Language-Model-Memorization-from-Tokenizer-Artifacts.git
cd Predicting-Language-Model-Memorization-from-Tokenizer-Artifacts

# Python env (pick one)
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install datasets   # for WikiText download in data scripts

# Optional: verify CUDA + PyTorch
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

Put large data and outputs on **scratch** if your site recommends it (replace paths below).

## 2. Smoke test (~15–45 min GPU, validates the stack)

Goal: prove imports, GPU, I/O, and the full pipeline work before a multi-hour job.

```bash
source .venv/bin/activate
cd /path/to/Predicting-Language-Model-Memorization-from-Tokenizer-Artifacts

# Generate WikiText + candidates (same as colab_real; ~3 min download)
python scripts/gen_real_data.py

# Fast HPC smoke config (small budget + few LM steps)
python -m src.run_pipeline --config configs/hpc_smoke.yaml
```

If this finishes with `PIPELINE COMPLETE`, your environment is good.

You can also use `bash scripts/hpc_smoke.sh` after editing paths inside it.

## 3. Full experiment (hours)

After smoke passes, run a larger config:

| Config | Use case |
|--------|----------|
| `configs/colab_max.yaml` | Strong single run (~3 h on T4-class GPU; faster on A100) |
| `configs/hpc_grid.yaml` | Large sweep — **requires** `data/raw/full_corpus.txt` and `data/candidates/canaries_full.jsonl` (you must create or symlink) |

```bash
python scripts/gen_max_data.py    # if using colab_max-style data
python -m src.run_pipeline --config configs/colab_max.yaml
```

For `hpc_grid.yaml`, prepare the corpus and canary files first, then:

```bash
python -m src.run_pipeline --config configs/hpc_grid.yaml
```

## 4. Batch scheduler (SLURM example)

Copy `scripts/slurm_template.sh`, set:

- `#SBATCH --partition=...`
- `#SBATCH --time=...` (walltime ≥ ref + target LM time + margin)
- `#SBATCH --gres=gpu:1`
- Project working directory and `venv` activation

Submit:

```bash
sbatch scripts/slurm_template.sh
```

## 5. After the run

- Main manifest: `results/<run_id>/<run_id>_pipeline.json`
- Metrics: `results/<run_id>/eval/<run_id>_eval_manifest.json` and `comparison.parquet`
- Copy `results/`, `labels/`, `features/`, `models/` to long-term storage if scratch is purged.

## 6. Troubleshooting

| Issue | What to check |
|-------|----------------|
| `CUDA not available` | On **login nodes** this is normal. Request a GPU (`srun`/`sbatch`), then test again. |
| `RuntimeError: No CUDA GPUs are available` | You called `torch.cuda.get_device_name(0)` on a **CPU-only** session. Use the **safe** check below — never call `get_device_name(0)` unless `is_available()` is True. |
| `Invalid job id specified` / stale `SLURM_JOB_ID` | Old `salloc` expired. `unset SLURM_JOB_ID` or open a **new SSH session**, then run `srun` again. |
| `salloc` says node ready but `hostname` is still `port` | You stay on the login node until you run `srun --pty bash` **inside** the allocation (or use `srun --gres=gpu:1 --time=... --pty bash` alone). |
| Job killed (OOM) | Lower `lm.training.batch_size` or `lm.d_model` / `n_layers` |
| Job killed (time) | Increase walltime or reduce `max_steps` / corpus budget |
| Missing files | Run `gen_*_data.py` for the config you use; check `corpus.base_source` and `canary.file` paths |

### Safe GPU check (copy-paste)

```bash
python -c "import torch; ok=torch.cuda.is_available(); print('cuda:', ok); print(torch.cuda.get_device_name(0) if ok else '(no GPU — use srun/sbatch on a compute node)')"
```

On **login node** you should see `cuda: False` with **no crash**. On a **GPU node**, `cuda: True` and a device name.

## 7. Edit `configs/hpc_grid.yaml` for your site

The template expects placeholder paths. Point `base_source` and `canary.file` to real files, or symlink WikiText + your canary JSONL into those paths.
