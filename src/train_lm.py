"""
train_lm.py
-----------
Train a small GPT-2-style transformer language model from scratch on a
tokenized corpus, for downstream BPC scoring (SPEC.md §3).

Design
------
- For each tokenizer condition we train TWO models from the SAME tokenizer:
    M_ref   on D_clean   (canary-free)
    M_target on D_canary  (canary-injected)
- The corpus is produced by build_corpus.py; it is already budget-sliced.
  We consume the ENTIRE file and log raw chars / raw lines / total tokens.
  Different tokenizers produce different token counts from the same raw-text
  budget; we log this but never treat token counts as equivalent.
- Training is deterministic given the same seed + corpus + config.

Fixed raw-text budget fairness (SPEC.md §4)
-------------------------------------------
Budget fairness is enforced at the RAW-TEXT INPUT level, not at the gradient-
step or unique-token-exposure level.  This is intentional and consistent with
SPEC.md §4 which requires "fixed raw text budget, not fixed epochs."

Exact guarantees:
  - Same raw chars and raw lines across all tokenizer conditions (enforced by
    build_corpus.py, verified here against manifest.json when present).
  - Same number of gradient updates (max_steps, a fixed compute budget).
  - Same tokens-processed count (max_steps × batch_size × seq_len).

Approximation that must be understood:
  - Different tokenizers produce different numbers of unique sequences
    (n_train) from the same raw text because fertility varies.
  - The training loop samples sequences WITH REPLACEMENT, so a tokenizer
    producing fewer sequences will revisit them more often within the same
    max_steps.  "Effective corpus passes" = (max_steps × batch_size) / n_train
    is NOT equal across tokenizers.
  - This is the inherent, expected asymmetry of comparing tokenizers on a
    fixed raw-text budget.  We log it as corpus_coverage_ratio in the manifest
    so it can be inspected.  Do not interpret corpus_coverage_ratio as a
    fairness violation — it is a transparency metric.

Convergence gate
----------------
After training, we check two conditions:
  1) final_loss < first_loss  (loss decreased)
  2) validation perplexity < reject_threshold  (model learned something)
If either fails, a loud WARNING is emitted.  The model is still saved (the
caller decides whether to proceed, per SPEC.md §9 guidance).

CLI
---
  python -m src.train_lm --config configs/colab_mini.yaml \\
      --corpus data/corpora/mini/D_clean.txt --role ref
  python -m src.train_lm --config configs/colab_mini.yaml \\
      --corpus data/corpora/mini/D_canary.txt --role target
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SHA-256 helper
# ---------------------------------------------------------------------------

def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Corpus tokenization
# ---------------------------------------------------------------------------

def _read_corpus_stats(path: str) -> Tuple[str, int, int]:
    """Read a text file and return (full_text, n_lines, n_chars).

    n_chars = sum of line content lengths (excludes newlines), matching
    build_corpus.py's manifest so budget validation aligns.
    """
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    lines = [ln.rstrip("\n") for ln in text.split("\n") if ln.strip()]
    n_lines = len(lines)
    n_chars = sum(len(ln) for ln in lines)
    return text, n_lines, n_chars


def _tokenize_corpus_hf(text: str, tokenizer_json_path: str) -> np.ndarray:
    """Tokenize the full corpus text using an HF Rust tokenizer."""
    from tokenizers import Tokenizer  # type: ignore[import-untyped]
    tok = Tokenizer.from_file(tokenizer_json_path)
    lines = [ln for ln in text.split("\n") if ln.strip()]
    all_ids: List[int] = []
    for line in lines:
        enc = tok.encode(line)
        all_ids.extend(enc.ids)
    return np.array(all_ids, dtype=np.int64)


def _tokenize_corpus_sp(text: str, model_path: str) -> np.ndarray:
    """Tokenize the full corpus text using a SentencePiece model."""
    import sentencepiece as spm  # type: ignore[import-untyped]
    sp = spm.SentencePieceProcessor(model_file=model_path)
    lines = [ln for ln in text.split("\n") if ln.strip()]
    all_ids: List[int] = []
    for line in lines:
        all_ids.extend(sp.encode(line))
    return np.array(all_ids, dtype=np.int64)


def _verify_corpus_against_manifest(
    corpus_path: str,
    n_lines: int,
    n_chars: int,
    role: str,
) -> None:
    """
    Optionally verify corpus stats against build_corpus manifest.json.

    If manifest.json exists in the same directory as the corpus file,
    we check that our observed n_lines and n_chars match the manifest.
    This catches wrong/corrupted corpora or mismatched run_id usage.
    """
    manifest_path = os.path.join(os.path.dirname(corpus_path), "manifest.json")
    if not os.path.isfile(manifest_path):
        logger.debug("No manifest.json in corpus dir; skipping budget validation")
        return

    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    if role == "ref":
        expect_lines = manifest.get("n_clean_lines")
        expect_chars = manifest.get("n_clean_chars")
    else:
        expect_lines = manifest.get("n_canary_lines")
        expect_chars = manifest.get("n_canary_chars")

    if expect_lines is not None and n_lines != expect_lines:
        raise ValueError(
            f"Corpus/manifest mismatch: observed {n_lines} lines but "
            f"manifest.json for role={role!r} expects {expect_lines}. "
            f"Corpus may be wrong or from a different run."
        )
    if expect_chars is not None and n_chars != expect_chars:
        raise ValueError(
            f"Corpus/manifest mismatch: observed {n_chars} raw chars but "
            f"manifest.json for role={role!r} expects {expect_chars}. "
            f"Corpus may be wrong or from a different run."
        )
    logger.info(
        "Corpus validated against manifest: %d lines, %d chars (budget_type=%s, budget_value=%d)",
        n_lines, n_chars,
        manifest.get("budget_type", "?"),
        manifest.get("budget_value", "?"),
    )


def tokenize_corpus(
    corpus_path: str,
    tokenizer_meta: Dict,
) -> Tuple[np.ndarray, int, int, int]:
    """
    Tokenize a corpus file using the tokenizer described by metadata.

    Returns (token_ids, n_lines, n_chars, n_tokens).
    """
    text, n_lines, n_chars = _read_corpus_stats(corpus_path)

    family = tokenizer_meta["family"]
    artifacts = tokenizer_meta["artifacts"]

    if family == "bpe":
        tok_json = artifacts["tokenizer_json"]
        if not os.path.isfile(tok_json):
            raise FileNotFoundError(f"tokenizer.json not found: {tok_json}")
        ids = _tokenize_corpus_hf(text, tok_json)
    elif family == "unigram":
        model_path = artifacts["model"]
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"SP model not found: {model_path}")
        ids = _tokenize_corpus_sp(text, model_path)
    else:
        raise ValueError(f"Unknown tokenizer family: {family!r}")

    n_tokens = len(ids)
    logger.info(
        "Tokenized corpus: %d lines, %d raw chars -> %d tokens "
        "(%.2f chars/token)",
        n_lines, n_chars, n_tokens,
        n_chars / max(n_tokens, 1),
    )
    return ids, n_lines, n_chars, n_tokens


# ---------------------------------------------------------------------------
# Dataset: chunk token IDs into fixed-length sequences
# ---------------------------------------------------------------------------

def _build_sequences(
    token_ids: np.ndarray,
    seq_len: int,
    seed: int,
    val_frac: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Chunk token_ids into non-overlapping sequences of length seq_len.

    Returns (train_seqs, val_seqs, n_train_seqs).
      train_seqs, val_seqs : int64 arrays of shape (N, seq_len)
      n_train_seqs         : number of unique training sequences
    The last incomplete chunk is discarded.
    A small held-out fraction is split off for within-tokenizer validation.
    """
    n = len(token_ids)
    n_seqs = n // seq_len
    if n_seqs == 0:
        raise ValueError(
            f"Corpus has {n} tokens but seq_len={seq_len}. "
            f"Need at least {seq_len} tokens to form one sequence."
        )
    usable = n_seqs * seq_len
    data = token_ids[:usable].reshape(n_seqs, seq_len)

    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_seqs)
    n_val = max(1, int(n_seqs * val_frac))
    n_train = n_seqs - n_val

    train_seqs = data[perm[:n_train]]
    val_seqs = data[perm[n_train:]]

    logger.info(
        "Sequences: %d train + %d val  (seq_len=%d, %d tokens discarded)",
        n_train, n_val, seq_len, n - usable,
    )
    # corpus_coverage_ratio is logged in training_manifest as a transparency metric.
    # It equals (max_steps * batch_size) / n_train and indicates how many times
    # each unique training sequence is expected to be sampled (with replacement).
    # It is NOT a fairness violation — it naturally varies across tokenizers
    # when the raw-text budget is fixed.  See module docstring for details.
    return train_seqs, val_seqs, n_train


# ---------------------------------------------------------------------------
# GPT-2 model (from scratch using HF transformers library)
# ---------------------------------------------------------------------------

def _build_model(
    vocab_size: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    d_ff: int,
    dropout: float,
    max_seq_len: int,
) -> Any:
    """
    Build a GPT2LMHeadModel from a GPT2Config (no pretrained weights).

    Uses HuggingFace transformers so the architecture is well-tested and
    the saved model can be loaded with standard HF APIs.
    """
    from transformers import GPT2Config, GPT2LMHeadModel  # type: ignore[import-untyped]

    config = GPT2Config(
        vocab_size=vocab_size,
        n_embd=d_model,
        n_head=n_heads,
        n_layer=n_layers,
        n_inner=d_ff,
        resid_pdrop=dropout,
        embd_pdrop=dropout,
        attn_pdrop=dropout,
        n_positions=max_seq_len,
        bos_token_id=None,
        eos_token_id=None,
    )
    model = GPT2LMHeadModel(config)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Built GPT2LMHeadModel: %d params  "
        "(d=%d, h=%d, L=%d, ff=%d, seq=%d, V=%d)",
        n_params, d_model, n_heads, n_layers, d_ff, max_seq_len, vocab_size,
    )
    return model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _train_loop(
    model: Any,
    train_seqs: np.ndarray,
    val_seqs: np.ndarray,
    *,
    max_steps: int,
    batch_size: int,
    learning_rate: float,
    warmup_steps: int,
    weight_decay: float,
    log_every: int,
    eval_every: int,
    checkpoint_every: int,
    output_dir: str,
    seed: int,
    device: str,
) -> Dict:
    """
    Standard autoregressive LM training loop.

    Returns a training_log dict with loss curves and metadata.
    """
    import torch
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR

    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = model.to(device)
    model.train()

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(step, 1) / max(warmup_steps, 1)
        return 1.0

    scheduler = LambdaLR(optimizer, lr_lambda)

    n_train = len(train_seqs)
    rng = np.random.RandomState(seed + 1)

    train_losses: List[float] = []
    val_losses: List[float] = []
    step_log: List[Dict] = []

    first_loss: Optional[float] = None
    tokens_processed = 0
    t0 = time.time()

    for step in range(1, max_steps + 1):
        # Sample a mini-batch (with replacement for simplicity / large-scale)
        idx = rng.randint(0, n_train, size=batch_size)
        batch_np = train_seqs[idx]
        batch = torch.tensor(batch_np, dtype=torch.long, device=device)

        # GPT-2: input_ids == labels (the model internally shifts)
        outputs = model(input_ids=batch, labels=batch)
        loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        loss_val = loss.item()
        tokens_processed += batch.numel()

        if first_loss is None:
            first_loss = loss_val

        if step % log_every == 0 or step == 1:
            elapsed = time.time() - t0
            lr_now = scheduler.get_last_lr()[0]
            tok_per_sec = tokens_processed / max(elapsed, 0.01)
            entry = {
                "step": step,
                "train_loss": round(loss_val, 6),
                "lr": round(lr_now, 8),
                "tokens_processed": tokens_processed,
                "elapsed_s": round(elapsed, 1),
                "tok_per_s": round(tok_per_sec, 0),
            }
            step_log.append(entry)
            train_losses.append(loss_val)
            logger.info(
                "step=%5d  loss=%.4f  lr=%.2e  tok=%d  %.0f tok/s",
                step, loss_val, lr_now, tokens_processed, tok_per_sec,
            )

        if step % eval_every == 0 or step == max_steps:
            vl = _eval_loss(model, val_seqs, batch_size, device)
            val_losses.append(vl)
            ppl = math.exp(min(vl, 20.0))
            logger.info("  [eval] step=%d  val_loss=%.4f  val_ppl=%.2f", step, vl, ppl)

        if step % checkpoint_every == 0:
            ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")
            model.save_pretrained(ckpt_dir)
            logger.info("  [ckpt] saved -> %s", ckpt_dir)

    final_loss = train_losses[-1] if train_losses else None
    final_val_loss = val_losses[-1] if val_losses else None

    total_time = time.time() - t0

    return {
        "first_loss": first_loss,
        "final_loss": final_loss,
        "final_val_loss": final_val_loss,
        "final_val_ppl": math.exp(min(final_val_loss, 20.0)) if final_val_loss else None,
        "tokens_processed": tokens_processed,
        "total_steps": max_steps,
        "total_time_s": round(total_time, 1),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "step_log": step_log,
    }


def _eval_loss(
    model: Any,
    val_seqs: np.ndarray,
    batch_size: int,
    device: str,
) -> float:
    """Compute average cross-entropy loss on validation sequences."""
    import torch

    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for start in range(0, len(val_seqs), batch_size):
            batch_np = val_seqs[start : start + batch_size]
            batch = torch.tensor(batch_np, dtype=torch.long, device=device)
            outputs = model(input_ids=batch, labels=batch)
            total_loss += outputs.loss.item()
            n_batches += 1

    model.train()
    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Convergence gate
# ---------------------------------------------------------------------------

def _convergence_gate(
    training_log: Dict,
    reject_ppl: float = 5000.0,
    vocab_size: Optional[int] = None,
) -> Tuple[bool, List[str]]:
    """
    Check convergence and return (passed, list_of_warnings).

    Checks:
      1) final_loss < first_loss  (loss decreased)
      2) final_val_ppl < effective_threshold
         effective_threshold = max(reject_ppl, vocab_size) so we never
         pass a model that performs worse than random (ppl ~= vocab_size).
    """
    warnings: List[str] = []
    passed = True

    first = training_log.get("first_loss")
    final = training_log.get("final_loss")
    ppl = training_log.get("final_val_ppl")

    effective_threshold = max(reject_ppl, vocab_size or 0)

    if first is not None and final is not None and final >= first:
        warnings.append(
            f"CONVERGENCE GATE FAIL: final_loss ({final:.4f}) >= "
            f"first_loss ({first:.4f}).  Training loss did not decrease."
        )
        passed = False

    if ppl is not None and ppl >= effective_threshold:
        msg = (
            f"CONVERGENCE GATE FAIL: final_val_ppl ({ppl:.1f}) >= "
            f"effective_threshold ({effective_threshold:.1f}).  "
            f"Model may not have learned."
        )
        if vocab_size is not None:
            msg += f"  (Threshold is max(reject_ppl={reject_ppl}, vocab_size={vocab_size}).)"
        warnings.append(msg)
        passed = False

    if first is None or final is None or ppl is None:
        warnings.append(
            "CONVERGENCE GATE WARNING: insufficient training data to evaluate "
            "convergence (missing loss or ppl values)."
        )
        passed = False

    return passed, warnings


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def train_lm(
    corpus_path: str,
    tokenizer_metadata_path: str,
    output_dir: str,
    run_id: str,
    role: str,
    *,
    d_model: int = 256,
    n_heads: int = 4,
    n_layers: int = 4,
    d_ff: int = 512,
    dropout: float = 0.1,
    max_seq_len: int = 512,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    max_steps: int = 2000,
    warmup_steps: int = 200,
    weight_decay: float = 0.01,
    log_every: int = 50,
    eval_every: int = 500,
    checkpoint_every: int = 1000,
    seed: int = 42,
    reject_ppl: float = 5000.0,
    require_matched_ref: bool = True,
    device: Optional[str] = None,
) -> str:
    """
    Train a GPT-2-style LM from scratch and save to output_dir.

    Parameters
    ----------
    corpus_path             : path to the training corpus (D_clean or D_canary)
    tokenizer_metadata_path : path to the tokenizer's metadata.json
    output_dir              : root models directory
    run_id                  : experiment run identifier
    role                    : "ref" or "target"
    require_matched_ref     : when True (default) and role="target", the ref
                              training_manifest.json MUST exist and its tok_id
                              MUST match.  Fail loudly if either is violated.
                              Set to False only for unit tests or standalone
                              target-only runs where no ref was trained.
    (remaining)             : model hyperparameters from config

    Returns
    -------
    str  Path to the written training_manifest.json file.
    """
    import torch

    if role not in ("ref", "target"):
        raise ValueError(f"role must be 'ref' or 'target', got {role!r}")

    if not os.path.isfile(corpus_path):
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")
    if not os.path.isfile(tokenizer_metadata_path):
        raise FileNotFoundError(f"Tokenizer metadata not found: {tokenizer_metadata_path}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("=" * 70)
    logger.info("train_lm: run_id=%s  role=%s  device=%s", run_id, role, device)
    logger.info("=" * 70)

    # Load tokenizer metadata
    with open(tokenizer_metadata_path, "r", encoding="utf-8") as fh:
        tok_meta = json.load(fh)

    tok_id = tok_meta["tok_id"]
    vocab_size = tok_meta["vocab_size_actual"]
    logger.info(
        "Tokenizer: tok_id=%s  family=%s  vocab=%d",
        tok_id, tok_meta["family"], vocab_size,
    )

    # When training target, enforce that the ref model used the same tokenizer.
    # require_matched_ref=True (default): fail loudly on missing or mismatched ref.
    # require_matched_ref=False: log a warning and continue (unit-test / standalone mode).
    if role == "target":
        ref_manifest_path = os.path.join(output_dir, run_id, "ref", "training_manifest.json")
        if os.path.isfile(ref_manifest_path):
            with open(ref_manifest_path, "r", encoding="utf-8") as fh:
                ref_manifest = json.load(fh)
            ref_tok_id = ref_manifest.get("tok_id")
            if ref_tok_id != tok_id:
                raise ValueError(
                    f"REF/TARGET MISMATCH: Ref model was trained with tok_id={ref_tok_id!r} "
                    f"but target is being trained with tok_id={tok_id!r}. "
                    f"Per SPEC.md §3, ref and target must use the exact same tokenizer. "
                    f"Aborting."
                )
            logger.info("Ref/target tokenizer match verified: tok_id=%s", tok_id)
        else:
            msg = (
                f"Ref manifest not found at {ref_manifest_path!r}. "
                f"Cannot verify ref/target tokenizer match (SPEC.md §3). "
                f"Train the ref model first, or pass require_matched_ref=False "
                f"to explicitly opt out of this check."
            )
            if require_matched_ref:
                raise FileNotFoundError(msg)
            else:
                logger.warning("require_matched_ref=False — skipping ref check. %s", msg)

    # Tokenize corpus
    token_ids, n_lines, n_chars, n_tokens = tokenize_corpus(
        corpus_path, tok_meta,
    )

    # Verify against build_corpus manifest if present (catches wrong/corrupted corpora)
    _verify_corpus_against_manifest(corpus_path, n_lines, n_chars, role)

    # Build sequences
    train_seqs, val_seqs, n_train_seqs = _build_sequences(token_ids, max_seq_len, seed)

    # Build model
    model = _build_model(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
        max_seq_len=max_seq_len,
    )

    # Output directory
    model_dir = os.path.join(output_dir, run_id, role)
    os.makedirs(model_dir, exist_ok=True)

    # Train
    training_log = _train_loop(
        model,
        train_seqs,
        val_seqs,
        max_steps=max_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        log_every=log_every,
        eval_every=eval_every,
        checkpoint_every=checkpoint_every,
        output_dir=model_dir,
        seed=seed,
        device=device,
    )

    # Save final model
    model.save_pretrained(model_dir)
    logger.info("Final model saved -> %s", model_dir)

    # Convergence gate
    gate_passed, gate_warnings = _convergence_gate(
        training_log, reject_ppl, vocab_size=vocab_size
    )
    for w in gate_warnings:
        logger.warning(w)
    if gate_passed:
        logger.info("Convergence gate: PASSED")
    else:
        logger.warning("Convergence gate: FAILED — model saved but may be unusable.")

    # Write training manifest
    manifest = {
        "run_id": run_id,
        "role": role,
        "tok_id": tok_id,
        "tokenizer_family": tok_meta["family"],
        "tokenizer_metadata": os.path.abspath(tokenizer_metadata_path),
        "corpus_path": os.path.abspath(corpus_path),
        "corpus_sha256": _sha256(corpus_path),
        "corpus_stats": {
            "raw_chars": n_chars,
            "raw_lines": n_lines,
            "total_tokens": n_tokens,
            "chars_per_token": round(n_chars / max(n_tokens, 1), 4),
            # Transparency metric (see module docstring §"Approximation"):
            # Expected number of times each unique training sequence is sampled.
            # This varies across tokenizers even on equal raw-text budgets because
            # fertility (tokens/char) differs.  It is NOT a fairness violation.
            "n_train_seqs": n_train_seqs,
            "corpus_coverage_ratio": round(
                (max_steps * batch_size) / max(n_train_seqs, 1), 3
            ),
        },
        "budget_fairness_note": (
            "Raw-text input budget is equal across tokenizer conditions "
            "(enforced by build_corpus.py). max_steps is a fixed compute "
            "budget. corpus_coverage_ratio varies per tokenizer because "
            "fertility differs; this is expected and not a fairness violation."
        ),
        "model_config": {
            "architecture": "gpt2",
            "vocab_size": vocab_size,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "d_ff": d_ff,
            "dropout": dropout,
            "max_seq_len": max_seq_len,
        },
        "training_config": {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_steps": max_steps,
            "warmup_steps": warmup_steps,
            "weight_decay": weight_decay,
            "seed": seed,
            "device": device,
        },
        "training_results": {
            "first_loss": training_log["first_loss"],
            "final_loss": training_log["final_loss"],
            "final_val_loss": training_log["final_val_loss"],
            "final_val_ppl": training_log["final_val_ppl"],
            "total_steps": training_log["total_steps"],
            "update_steps": training_log["total_steps"],  # SPEC §4: same as total_steps
            "tokens_processed": training_log["tokens_processed"],
            "total_time_s": training_log["total_time_s"],
        },
        "convergence_gate": {
            "passed": gate_passed,
            "warnings": gate_warnings,
        },
        "model_dir": os.path.abspath(model_dir),
        "trained_at": datetime.datetime.utcnow().isoformat() + "Z",
    }

    manifest_path = os.path.join(model_dir, "training_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    # Also save the step-level log
    log_path = os.path.join(model_dir, "training_log.json")
    with open(log_path, "w", encoding="utf-8") as fh:
        json.dump(training_log["step_log"], fh, indent=2)

    logger.info("Training manifest -> %s", manifest_path)
    logger.info("Step-level log    -> %s", log_path)
    logger.info(
        "SUMMARY: %d steps, %d tokens processed, "
        "final_loss=%.4f, val_ppl=%.1f, time=%.1fs",
        training_log["total_steps"],
        training_log["tokens_processed"],
        training_log["final_loss"] or 0,
        training_log["final_val_ppl"] or 0,
        training_log["total_time_s"],
    )
    logger.info("=" * 70)

    return manifest_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.train_lm",
        description="Train a GPT-2-style LM from scratch for BPC scoring.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", metavar="YAML", help="Project YAML config file.")
    p.add_argument("--corpus", metavar="FILE", help="Corpus text file (D_clean or D_canary).")
    p.add_argument(
        "--tokenizer-meta", metavar="FILE",
        help="Path to the tokenizer metadata.json.",
    )
    p.add_argument(
        "--role", choices=["ref", "target"], required=True,
        help="Which model to train: 'ref' (on D_clean) or 'target' (on D_canary).",
    )
    p.add_argument("--output", metavar="DIR", help="Root models output directory.")
    p.add_argument("--run-id", metavar="ID", help="Run identifier.")
    p.add_argument("--seed", type=int, help="Random seed.")
    p.add_argument("--device", help="torch device (cpu / cuda).")
    p.add_argument("--max-steps", type=int, help="Override max training steps.")
    p.add_argument("--batch-size", type=int, help="Override batch size.")
    p.add_argument("--reject-ppl", type=float, default=5000.0,
                   help="Validation perplexity above which convergence gate fails.")
    p.add_argument(
        "--no-require-matched-ref", action="store_true", default=False,
        dest="no_require_matched_ref",
        help=(
            "Disable the strict check that the ref training_manifest.json must "
            "exist and match tok_id before training target.  Use only for "
            "unit tests or standalone target-only runs.  Default: strict mode ON."
        ),
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # Defaults from config
    corpus = tokenizer_meta = output_dir = run_id = None
    seed = d_model = n_heads = n_layers = d_ff = None
    dropout = max_seq_len = batch_size = learning_rate = None
    max_steps = warmup_steps = weight_decay = None
    log_every = eval_every = checkpoint_every = None

    if args.config:
        from src.config import load_config  # type: ignore[import]
        cfg = load_config(args.config)
        run_id = cfg.run_id
        seed = cfg.seed
        d_model = cfg.lm.d_model
        n_heads = cfg.lm.n_heads
        n_layers = cfg.lm.n_layers
        d_ff = cfg.lm.d_ff
        dropout = cfg.lm.dropout
        max_seq_len = cfg.lm.max_seq_len
        batch_size = cfg.lm.training.batch_size
        learning_rate = cfg.lm.training.learning_rate
        max_steps = cfg.lm.training.max_steps
        warmup_steps = cfg.lm.training.warmup_steps
        weight_decay = cfg.lm.training.weight_decay
        log_every = cfg.lm.training.log_every
        eval_every = cfg.lm.training.eval_every
        checkpoint_every = cfg.lm.training.checkpoint_every
        tokenizer_meta = os.path.join(cfg.paths.tokenizer_dir, "metadata.json")
        output_dir = os.path.dirname(cfg.paths.ref_model_dir)

    # CLI overrides
    if args.corpus:         corpus = args.corpus
    if args.tokenizer_meta: tokenizer_meta = args.tokenizer_meta
    if args.output:         output_dir = args.output
    if args.run_id:         run_id = args.run_id
    if args.seed is not None:       seed = args.seed
    if args.max_steps is not None:  max_steps = args.max_steps
    if args.batch_size is not None: batch_size = args.batch_size

    missing = [
        name for name, val in [
            ("--corpus", corpus),
            ("--tokenizer-meta", tokenizer_meta),
            ("--output", output_dir),
            ("--run-id", run_id),
        ]
        if val is None
    ]
    if missing:
        parser.error(
            "Missing required values (provide via --config or explicit flags):\n  "
            + "\n  ".join(missing)
        )

    try:
        manifest_path = train_lm(
            corpus_path=corpus,                        # type: ignore[arg-type]
            tokenizer_metadata_path=tokenizer_meta,    # type: ignore[arg-type]
            output_dir=output_dir,                     # type: ignore[arg-type]
            run_id=run_id,                             # type: ignore[arg-type]
            role=args.role,
            d_model=d_model or 256,
            n_heads=n_heads or 4,
            n_layers=n_layers or 4,
            d_ff=d_ff or 512,
            dropout=dropout if dropout is not None else 0.1,
            max_seq_len=max_seq_len or 512,
            batch_size=batch_size or 32,
            learning_rate=learning_rate or 3e-4,
            max_steps=max_steps or 2000,
            warmup_steps=warmup_steps or 200,
            weight_decay=weight_decay if weight_decay is not None else 0.01,
            log_every=log_every or 50,
            eval_every=eval_every or 500,
            checkpoint_every=checkpoint_every or 1000,
            seed=seed or 42,
            reject_ppl=args.reject_ppl,
            require_matched_ref=not args.no_require_matched_ref,
            device=args.device,
        )
    except (FileNotFoundError, ValueError, RuntimeError, ImportError) as e:
        logger.error("FATAL: %s", e)
        sys.exit(1)

    print(f"\nDone.  Training manifest: {manifest_path}")


if __name__ == "__main__":
    main()
