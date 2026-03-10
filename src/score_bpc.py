"""
score_bpc.py
------------
Score candidate strings with bits-per-character (BPC) from a trained LM.

Implements SPEC.md §2.3 and §2.4 exactly:

  tokens     = Tok.encode(Normalize_Tok(s))     — x_1 ... x_T
  total_bits = - sum_{t=1..T-1} log2 p(x_{t+1} | x_{<=t})
  bpc        = total_bits / len(Normalize_Tok(s))

Token-level scoring mechanics
-----------------------------
- GPT2LMHeadModel.forward(input_ids=X, labels=X) computes per-position
  cross-entropy in **nats** (natural log) then returns the **mean** over
  the T-1 valid prediction positions (it shifts labels internally).
- We recover total bits as:
      total_bits = mean_nats * (T - 1) / ln(2)
  The / ln(2) converts nats to bits.  The * (T-1) un-averages.

BOS/EOS policy (SPEC.md §2.3 default)
--------------------------------------
- The models in this pipeline are trained without explicit BOS/EOS tokens.
- We score exactly the token sequence produced by Tok.encode(s_norm).
- No BOS or EOS tokens are prepended or appended.
- add_bos and add_eos config flags exist but default to False; if set True,
  the code logs a warning and overrides to False for consistency with
  training (SPEC.md §2.3).

Edge case validity policies
----------------------------
  Condition               | valid | invalid_reason            | total_bits | bpc
  ------------------------|-------|---------------------------|------------|-----
  empty_after_norm        | False | empty_after_normalization | NaN        | NaN
  zero_tokens             | False | zero_tokens               | NaN        | NaN
  T < 2 (fewer than 2)   | False | fewer_than_2_tokens       | NaN        | NaN
  truncated (default)     | False | truncated                 | NaN        | NaN
  truncated (allow=True)  | True  | None                      | scored     | scored

Rationale for T<2 invalidity: with a single token there are zero
autoregressive prediction positions, so total_bits is undefined (not zero).
Assigning 0 would make the string look maximally "easy" and contaminate
downstream ΔBPC labels.

Truncation policy (config-controlled)
-------------------------------------
- allow_truncation (default: false) controls whether strings exceeding
  max_seq_len are scored on their prefix or marked invalid.
- When false (strict, recommended for real runs): truncated rows get
  valid=False, invalid_reason="truncated", total_bits/bpc=NaN.
- When true (lenient, useful for exploration): truncated rows are scored
  on the prefix, with truncated=True flagged and a warning logged.

CLI
---
  python -m src.score_bpc \\
      --model-dir models/colab_mini/ref \\
      --tokenizer-meta tokenizers/mini_bpe/metadata.json \\
      --candidates data/candidates/canaries_mini.jsonl \\
      --run-id colab_mini --role ref --output labels/

  python -m src.score_bpc --config configs/colab_mini.yaml --role ref
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_NAN = float("nan")
_LN2 = math.log(2.0)


# ---------------------------------------------------------------------------
# Tokenizer loading (reuse extract_features infrastructure)
# ---------------------------------------------------------------------------

def _load_tokenizer_obj(metadata_path: str) -> Tuple[Any, Dict, str]:
    """
    Load the raw tokenizer object and its metadata.

    Returns (tok_obj, metadata_dict, family).
    tok_obj is a tokenizers.Tokenizer (BPE) or spm.SentencePieceProcessor (unigram).
    """
    metadata_path = os.path.abspath(metadata_path)
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"Tokenizer metadata not found: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)

    for field in ("tok_id", "family", "vocab_size_actual", "artifacts"):
        if field not in meta:
            raise ValueError(
                f"metadata.json at {metadata_path!r} missing field {field!r}"
            )

    family = meta["family"]
    artifacts = meta["artifacts"]

    if family == "bpe":
        from tokenizers import Tokenizer  # type: ignore[import-untyped]
        tok_json = artifacts.get("tokenizer_json")
        if not tok_json or not os.path.isfile(tok_json):
            raise FileNotFoundError(
                f"tokenizer.json not found: {tok_json!r} (from {metadata_path!r})"
            )
        tok_obj = Tokenizer.from_file(tok_json)

    elif family == "unigram":
        import sentencepiece as spm  # type: ignore[import-untyped]
        model_path = artifacts.get("model")
        if not model_path or not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"SP .model not found: {model_path!r} (from {metadata_path!r})"
            )
        tok_obj = spm.SentencePieceProcessor(model_file=model_path)

    else:
        raise ValueError(f"Unknown tokenizer family {family!r} in {metadata_path!r}")

    logger.info(
        "Loaded tokenizer: tok_id=%s  family=%s  vocab=%d",
        meta["tok_id"], family, meta["vocab_size_actual"],
    )
    return tok_obj, meta, family


def _make_encode_fn(
    tok_obj: Any, family: str
) -> Callable[[str], List[int]]:
    """Return a function str -> List[int] for the given tokenizer."""
    if family == "bpe":
        def _encode(s: str) -> List[int]:
            return tok_obj.encode(s).ids
    elif family == "unigram":
        def _encode(s: str) -> List[int]:
            return tok_obj.encode(s)  # type: ignore[return-value]
    else:
        raise ValueError(f"Unsupported family: {family!r}")
    return _encode


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model(model_dir: str, device: str) -> Any:
    """Load a GPT2LMHeadModel from a directory saved by model.save_pretrained()."""
    import torch
    from transformers import GPT2LMHeadModel  # type: ignore[import-untyped]

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    model = GPT2LMHeadModel.from_pretrained(model_dir)
    model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    max_pos = model.config.n_positions
    logger.info(
        "Loaded model: %d params, max_seq_len=%d, device=%s",
        n_params, max_pos, device,
    )
    return model


# ---------------------------------------------------------------------------
# Candidate file loading (reuse from extract_features)
# ---------------------------------------------------------------------------

def load_candidates(path: str) -> List[Dict]:
    """Load JSONL candidates with at least 'id' and 'text' fields."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Candidates file not found: {path}")
    candidates = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            if not isinstance(obj, dict):
                raise ValueError(f"Line {lineno}: expected JSON object")
            for f in ("id", "text"):
                if f not in obj:
                    raise ValueError(f"Line {lineno}: missing field {f!r}")
            candidates.append(obj)

    seen = set()
    for c in candidates:
        if c["id"] in seen:
            raise ValueError(f"Duplicate candidate id: {c['id']!r}")
        seen.add(c["id"])

    if not candidates:
        raise ValueError(f"Candidates file {path!r} is empty.")

    logger.info("Loaded %d candidates from %s", len(candidates), path)
    return candidates


# ---------------------------------------------------------------------------
# Core scoring: per-string total_bits computation
# ---------------------------------------------------------------------------

def _score_single(
    token_ids: List[int],
    model: Any,
    device: str,
    max_seq_len: int,
) -> Tuple[float, bool]:
    """
    Compute total negative log-likelihood in bits for a single token sequence.

    Parameters
    ----------
    token_ids  : list of int, length T
    model      : GPT2LMHeadModel in eval mode
    device     : torch device string
    max_seq_len: model's n_positions limit

    Returns
    -------
    (total_bits, was_truncated)

    Math (SPEC.md §2.3):
      total_bits = - sum_{t=1..T-1} log2 p(x_{t+1} | x_{<=t})

    GPT2LMHeadModel.forward(input_ids=X, labels=X) returns:
      loss = mean of per-position cross-entropy in nats over T-1 positions
      (it internally shifts labels by 1)

    Conversion:
      total_nats = loss.item() * (T - 1)
      total_bits = total_nats / ln(2)
    """
    import torch

    T = len(token_ids)
    if T < 2:
        return _NAN, False

    truncated = False
    if T > max_seq_len:
        token_ids = token_ids[:max_seq_len]
        T = max_seq_len
        truncated = True

    ids_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(input_ids=ids_tensor, labels=ids_tensor)

    mean_nats = outputs.loss.item()
    n_predictions = T - 1
    total_bits = mean_nats * n_predictions / _LN2

    return total_bits, truncated


def _score_batch(
    batch_ids: List[List[int]],
    model: Any,
    device: str,
    max_seq_len: int,
) -> List[Tuple[float, bool]]:
    """
    Score a batch of token sequences, padding to the longest.

    Shorter sequences are right-padded with 0 and masked via labels=-100
    so padding tokens don't contribute to the loss.

    Returns one (total_bits, was_truncated) per sequence.
    """
    import torch

    results: List[Tuple[float, bool]] = []
    if not batch_ids:
        return results

    truncated_flags = []
    effective_ids = []
    for ids in batch_ids:
        trunc = False
        if len(ids) > max_seq_len:
            ids = ids[:max_seq_len]
            trunc = True
        truncated_flags.append(trunc)
        effective_ids.append(ids)

    lengths = [len(ids) for ids in effective_ids]
    max_len = max(lengths)

    # Build padded input_ids and labels tensors
    pad_id = 0
    ignore_index = -100

    input_batch = []
    label_batch = []
    for ids, L in zip(effective_ids, lengths):
        padded = ids + [pad_id] * (max_len - L)
        labels = list(ids) + [ignore_index] * (max_len - L)
        input_batch.append(padded)
        label_batch.append(labels)

    input_tensor = torch.tensor(input_batch, dtype=torch.long, device=device)
    label_tensor = torch.tensor(label_batch, dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(input_ids=input_tensor, labels=label_tensor)

    # outputs.loss is the MEAN across all non-ignored positions in the BATCH.
    # We need per-sequence totals, so we recompute from logits.
    logits = outputs.logits  # (B, max_len, V)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # nats

    for i, (ids, L, trunc) in enumerate(zip(effective_ids, lengths, truncated_flags)):
        if L < 2:
            results.append((_NAN, trunc))
            continue

        # Per-position NLL: predict token t+1 from position t
        # Shift: logits at positions 0..L-2 predict tokens at positions 1..L-1
        seq_log_probs = log_probs[i]  # (max_len, V)
        target_ids = torch.tensor(ids[1:L], dtype=torch.long, device=device)
        pred_log_probs = seq_log_probs[:L - 1]  # (L-1, V)
        token_nll_nats = -pred_log_probs[
            torch.arange(L - 1, device=device), target_ids
        ]  # (L-1,)

        total_nats = token_nll_nats.sum().item()
        total_bits = total_nats / _LN2
        results.append((total_bits, trunc))

    return results


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def score_bpc(
    model_dir: str,
    tokenizer_metadata_path: str,
    candidates_path: str,
    output_dir: str,
    run_id: str,
    role: str,
    *,
    batch_size: int = 64,
    add_bos: bool = False,
    add_eos: bool = False,
    allow_truncation: bool = False,
    device: Optional[str] = None,
) -> str:
    """
    Score all candidate strings with BPC from a trained model.

    Parameters
    ----------
    model_dir              : path to saved GPT2LMHeadModel directory
    tokenizer_metadata_path: path to tokenizer metadata.json
    candidates_path        : JSONL file with {id, text} per line
    output_dir             : directory for output Parquet file
    run_id                 : experiment run identifier
    role                   : "ref" or "target"
    batch_size             : number of sequences to score at once
    add_bos / add_eos      : whether to prepend/append special tokens (default: False)
    allow_truncation       : if False (default), strings exceeding max_seq_len are
                             marked invalid.  If True, they are truncated and scored
                             with a warning.
    device                 : torch device

    Returns
    -------
    str  Path to the written Parquet file.
    """
    import torch

    if role not in ("ref", "target"):
        raise ValueError(f"role must be 'ref' or 'target', got {role!r}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("=" * 65)
    logger.info("score_bpc: run_id=%s  role=%s  device=%s", run_id, role, device)
    logger.info("=" * 65)

    # Load tokenizer
    tok_obj, tok_meta, family = _load_tokenizer_obj(tokenizer_metadata_path)
    tok_id = tok_meta["tok_id"]
    encode_fn = _make_encode_fn(tok_obj, family)

    # Get normalizer
    from src.normalize import get_normalizer_fn  # type: ignore[import]
    norm_fn, norm_id = get_normalizer_fn(tok_obj)
    logger.info("Normalizer: %s", norm_id)

    # BOS/EOS policy: SPEC.md §2.3 default is to score exactly what
    # Tok.encode(s_norm) returns, without adding special tokens.
    # If the model was trained without explicit BOS/EOS (the default for
    # this pipeline), adding them at scoring time would be inconsistent.
    if add_bos:
        logger.warning(
            "add_bos=True requested, but the model in this pipeline was "
            "trained without an explicit BOS token (bos_token_id=None). "
            "Adding BOS at scoring time would be inconsistent with training. "
            "Overriding to add_bos=False per SPEC.md §2.3 default."
        )
        add_bos = False
    if add_eos:
        logger.warning(
            "add_eos=True requested, but the model in this pipeline was "
            "trained without an explicit EOS token (eos_token_id=None). "
            "Adding EOS at scoring time would be inconsistent with training. "
            "Overriding to add_eos=False per SPEC.md §2.3 default."
        )
        add_eos = False

    logger.info(
        "Scoring policy: add_bos=%s  add_eos=%s  allow_truncation=%s",
        add_bos, add_eos, allow_truncation,
    )

    # Load model
    model = _load_model(model_dir, device)
    max_seq_len = model.config.n_positions

    # Verify tokenizer/model consistency via training manifest
    manifest_path = os.path.join(model_dir, "training_manifest.json")
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as fh:
            train_manifest = json.load(fh)
        manifest_tok_id = train_manifest.get("tok_id")
        if manifest_tok_id and manifest_tok_id != tok_id:
            raise ValueError(
                f"Tokenizer/model mismatch: model trained with tok_id={manifest_tok_id!r} "
                f"but scoring with tok_id={tok_id!r}. Per SPEC.md §3, these must match."
            )
        manifest_vocab = train_manifest.get("model_config", {}).get("vocab_size")
        if manifest_vocab and manifest_vocab != tok_meta["vocab_size_actual"]:
            raise ValueError(
                f"Vocab size mismatch: model trained with vocab_size={manifest_vocab} "
                f"but tokenizer has vocab_size={tok_meta['vocab_size_actual']}."
            )
        logger.info("Model/tokenizer consistency verified against training_manifest.json")
    else:
        logger.warning(
            "No training_manifest.json in %s; cannot verify tokenizer/model match.",
            model_dir,
        )

    # Load candidates
    candidates = load_candidates(candidates_path)
    n = len(candidates)

    # Normalize and tokenize all candidates
    logger.info("Normalizing and tokenizing %d candidates...", n)
    records: List[Dict] = []
    for cand in candidates:
        cid = cand["id"]
        text_raw = cand["text"]
        s_norm = norm_fn(text_raw)
        len_chars = len(s_norm)

        if not s_norm:
            logger.warning(
                "candidate id=%s: empty after normalization, marking invalid.", cid
            )
            records.append({
                "candidate_id": cid,
                "text_raw": text_raw,
                "normalized_text": s_norm,
                "len_chars": 0,
                "n_tokens": 0,
                "token_ids": [],
                "valid": False,
                "invalid_reason": "empty_after_normalization",
            })
            continue

        ids = encode_fn(s_norm)
        n_tokens = len(ids)

        if n_tokens == 0:
            logger.warning(
                "candidate id=%s: tokenizer produced 0 tokens, marking invalid.", cid
            )
            records.append({
                "candidate_id": cid,
                "text_raw": text_raw,
                "normalized_text": s_norm,
                "len_chars": len_chars,
                "n_tokens": 0,
                "token_ids": [],
                "valid": False,
                "invalid_reason": "zero_tokens",
            })
            continue

        if n_tokens < 2:
            logger.warning(
                "candidate id=%s: %d token(s) — fewer than 2, no autoregressive "
                "prediction positions exist. Marking invalid.", cid, n_tokens
            )
            records.append({
                "candidate_id": cid,
                "text_raw": text_raw,
                "normalized_text": s_norm,
                "len_chars": len_chars,
                "n_tokens": n_tokens,
                "token_ids": ids,
                "valid": False,
                "invalid_reason": "fewer_than_2_tokens",
            })
            continue

        records.append({
            "candidate_id": cid,
            "text_raw": text_raw,
            "normalized_text": s_norm,
            "len_chars": len_chars,
            "n_tokens": n_tokens,
            "token_ids": ids,
            "valid": True,
            "invalid_reason": None,
        })

    # Score in batches
    logger.info("Scoring %d candidates in batches of %d...", n, batch_size)

    # Collect indices of valid records that need scoring
    to_score_indices = [i for i, r in enumerate(records) if r["valid"]]
    n_valid = len(to_score_indices)
    n_scored = 0

    for batch_start in range(0, n_valid, batch_size):
        batch_indices = to_score_indices[batch_start : batch_start + batch_size]
        batch_ids = [records[i]["token_ids"] for i in batch_indices]

        results = _score_batch(batch_ids, model, device, max_seq_len)

        for idx, (total_bits, truncated) in zip(batch_indices, results):
            records[idx]["truncated"] = truncated

            if truncated and not allow_truncation:
                logger.warning(
                    "candidate id=%s: truncated from %d to %d tokens; "
                    "marking invalid (allow_truncation=False).",
                    records[idx]["candidate_id"],
                    len(records[idx]["token_ids"]),
                    max_seq_len,
                )
                records[idx]["total_bits"] = _NAN
                records[idx]["bpc"] = _NAN
                records[idx]["valid"] = False
                records[idx]["invalid_reason"] = "truncated"
                continue

            if truncated:
                logger.warning(
                    "candidate id=%s: truncated from %d to %d tokens; "
                    "scoring prefix only (allow_truncation=True).",
                    records[idx]["candidate_id"],
                    len(records[idx]["token_ids"]),
                    max_seq_len,
                )

            records[idx]["total_bits"] = total_bits
            len_chars = records[idx]["len_chars"]
            if len_chars > 0 and not math.isnan(total_bits):
                records[idx]["bpc"] = total_bits / len_chars
            else:
                records[idx]["bpc"] = _NAN

        n_scored += len(batch_indices)
        if n_scored % 500 == 0 or batch_start + batch_size >= n_valid:
            logger.info("  scored %d / %d", n_scored, n_valid)

    # Fill invalid records
    for r in records:
        if not r["valid"]:
            r.setdefault("total_bits", _NAN)
            r.setdefault("bpc", _NAN)
            r.setdefault("truncated", False)

    # Build output rows (drop token_ids — large and not needed in output)
    import pandas as pd  # type: ignore[import-untyped]

    rows = []
    for r in records:
        rows.append({
            "candidate_id": r["candidate_id"],
            "text_raw": r["text_raw"],
            "normalized_text": r["normalized_text"],
            "len_chars": r["len_chars"],
            "n_tokens": r["n_tokens"],
            "total_bits": r["total_bits"],
            "bpc": r["bpc"],
            "valid": r["valid"],
            "invalid_reason": r.get("invalid_reason"),
            "truncated": r["truncated"],
            "tok_id": tok_id,
            "normalizer_id": norm_id,
            "role": role,
            "run_id": run_id,
            "model_dir": os.path.abspath(model_dir),
            "scored_at": datetime.datetime.utcnow().isoformat() + "Z",
        })

    df = pd.DataFrame(rows)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{run_id}_{role}_bpc.parquet")
    df.to_parquet(out_path, index=False, engine="pyarrow")

    n_final_valid = sum(1 for r in records if r["valid"])
    n_final_invalid = sum(1 for r in records if not r["valid"])
    n_trunc = sum(1 for r in records if r.get("truncated"))
    logger.info(
        "Wrote %d rows (%d valid, %d invalid, %d truncated) -> %s",
        len(df), n_final_valid, n_final_invalid, n_trunc, out_path,
    )
    logger.info("=" * 65)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.score_bpc",
        description="Score candidate strings with BPC from a trained LM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", metavar="YAML", help="Project YAML config.")
    p.add_argument("--model-dir", metavar="DIR", help="Saved model directory.")
    p.add_argument("--tokenizer-meta", metavar="FILE", help="Tokenizer metadata.json.")
    p.add_argument("--candidates", metavar="FILE", help="Candidates JSONL file.")
    p.add_argument("--role", choices=["ref", "target"], required=True)
    p.add_argument("--run-id", metavar="ID")
    p.add_argument("--output", metavar="DIR", help="Output directory.")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--device", help="torch device.")
    p.add_argument(
        "--allow-truncation", action="store_true", default=False,
        help="If set, truncated strings remain valid (scored on prefix). "
             "Default: truncated strings are marked invalid.",
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

    model_dir = tokenizer_meta = candidates = output_dir = run_id = None
    batch_size: Optional[int] = None
    add_bos = add_eos = False
    allow_truncation = False

    if args.config:
        from src.config import load_config  # type: ignore[import]
        cfg = load_config(args.config)
        run_id = cfg.run_id
        tokenizer_meta = os.path.join(cfg.paths.tokenizer_dir, "metadata.json")
        if args.role == "ref":
            model_dir = cfg.paths.ref_model_dir
        else:
            model_dir = cfg.paths.target_model_dir
        candidates = cfg.corpus.canary.file
        output_dir = cfg.paths.labels_dir
        batch_size = cfg.scoring.batch_size
        add_bos = cfg.scoring.add_bos
        add_eos = cfg.scoring.add_eos
        allow_truncation = cfg.scoring.allow_truncation

    if args.model_dir:       model_dir = args.model_dir
    if args.tokenizer_meta:  tokenizer_meta = args.tokenizer_meta
    if args.candidates:      candidates = args.candidates
    if args.output:          output_dir = args.output
    if args.run_id:          run_id = args.run_id
    if args.batch_size:      batch_size = args.batch_size
    if args.allow_truncation: allow_truncation = True

    missing = [
        name for name, val in [
            ("--model-dir", model_dir),
            ("--tokenizer-meta", tokenizer_meta),
            ("--candidates", candidates),
            ("--output", output_dir),
            ("--run-id", run_id),
        ]
        if val is None
    ]
    if missing:
        parser.error("Missing:\n  " + "\n  ".join(missing))

    try:
        out = score_bpc(
            model_dir=model_dir,             # type: ignore[arg-type]
            tokenizer_metadata_path=tokenizer_meta,  # type: ignore[arg-type]
            candidates_path=candidates,      # type: ignore[arg-type]
            output_dir=output_dir,           # type: ignore[arg-type]
            run_id=run_id,                   # type: ignore[arg-type]
            role=args.role,
            batch_size=batch_size or 64,
            add_bos=add_bos,
            add_eos=add_eos,
            allow_truncation=allow_truncation,
            device=args.device,
        )
    except (FileNotFoundError, ValueError, RuntimeError, ImportError) as e:
        logger.error("FATAL: %s", e)
        sys.exit(1)

    print(f"\nDone.  Scores written to: {out}")


if __name__ == "__main__":
    main()
