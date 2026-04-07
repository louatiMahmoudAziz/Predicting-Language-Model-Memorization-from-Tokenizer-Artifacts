"""
pretrained_eval.py
------------------
Evaluate memorization risk prediction on REAL pretrained language models
(Pythia, GPT-2, GPT-Neo) using their native tokenizers.

This module addresses the key scientific limitation of the canary-injection
pipeline: it tests whether tokenizer-derived features predict memorization
in models where the evaluator does NOT control the training data.

Ground truth: a string is "memorized" if its perplexity under a LARGER model
is significantly lower than under a SMALLER model from the same family
(cross-scale memorization signal), OR if it appears verbatim in known training
data (The Pile) and has anomalously low BPC.

Three complementary memorization signals are supported:
  1. Absolute BPC:  low BPC under the model (string is easy for the model)
  2. Cross-scale delta:  BPC(small) - BPC(large) >> 0  (large model "knows" it)
  3. zlib ratio:  zlib_bits(s) / model_bits(s) >> 1  (model is suspiciously good)

Usage:
    python -m src.pretrained_eval \\
        --model EleutherAI/pythia-1.4b \\
        --ref-model EleutherAI/pythia-70m \\
        --candidates data/candidates/pile_candidates.jsonl \\
        --output results/pythia_1b/ \\
        --run-id pythia_1b_eval
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
import zlib
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# zlib baseline
# ---------------------------------------------------------------------------

def zlib_bits(text: str) -> float:
    """Compute the number of bits to represent `text` via zlib compression."""
    encoded = text.encode("utf-8")
    compressed = zlib.compress(encoded, level=9)
    return len(compressed) * 8.0


def zlib_bpc(text: str) -> float:
    """Bits-per-character via zlib compression."""
    if not text:
        return float("nan")
    return zlib_bits(text) / len(text)


# ---------------------------------------------------------------------------
# Model loader — pretrained HuggingFace models
# ---------------------------------------------------------------------------

def load_pretrained_model(model_name: str, device: str = "auto"):
    """
    Load a pretrained causal LM and its tokenizer from HuggingFace.

    Returns (model, tokenizer, device_str).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading pretrained model: %s", model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    logger.info("Using device: %s", device)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Loaded %s: %.1fM params, vocab_size=%d, device=%s",
        model_name, n_params / 1e6, len(tokenizer), device,
    )

    return model, tokenizer, device


# ---------------------------------------------------------------------------
# BPC scoring for pretrained models
# ---------------------------------------------------------------------------

def score_pretrained_bpc(
    model: Any,
    tokenizer: Any,
    texts: List[str],
    device: str,
    batch_size: int = 32,
    max_length: int = 1024,
) -> List[Dict[str, Any]]:
    """
    Compute BPC for each text under a pretrained model.

    Returns a list of dicts with keys:
      - text_raw, n_tokens, total_bits, bpc, valid, truncated
    """
    import torch

    results = []
    model.eval()

    for batch_start in range(0, len(texts), batch_size):
        batch_texts = texts[batch_start:batch_start + batch_size]

        for text in batch_texts:
            if not text.strip():
                results.append({
                    "text_raw": text,
                    "n_tokens": 0,
                    "total_bits": float("nan"),
                    "bpc": float("nan"),
                    "valid": False,
                    "truncated": False,
                })
                continue

            # Tokenize
            encoding = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
            )
            input_ids = encoding["input_ids"].to(device)
            n_tokens = input_ids.shape[1]

            truncated = n_tokens >= max_length
            if n_tokens < 2:
                results.append({
                    "text_raw": text,
                    "n_tokens": n_tokens,
                    "total_bits": float("nan"),
                    "bpc": float("nan"),
                    "valid": False,
                    "truncated": truncated,
                })
                continue

            # Forward pass — teacher-forced
            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
                # outputs.loss is mean cross-entropy in nats over (T-1) positions
                mean_nats = outputs.loss.item()

            # Convert: total_bits = mean_nats * (T-1) / ln(2)
            T = n_tokens
            total_bits = mean_nats * (T - 1) / math.log(2)

            # BPC = total_bits / len(text)
            # Use character length of original text (not tokens)
            char_len = len(text)
            bpc = total_bits / char_len if char_len > 0 else float("nan")

            results.append({
                "text_raw": text,
                "n_tokens": n_tokens,
                "total_bits": total_bits,
                "bpc": bpc,
                "valid": True,
                "truncated": truncated,
            })

        if (batch_start + batch_size) % 200 == 0 or (batch_start + batch_size) >= len(texts):
            logger.info(
                "Scored %d / %d texts",
                min(batch_start + batch_size, len(texts)), len(texts),
            )

    return results


# ---------------------------------------------------------------------------
# Feature extraction for pretrained tokenizers
# ---------------------------------------------------------------------------

def extract_pretrained_features(
    tokenizer: Any,
    texts: List[str],
    candidate_ids: List[str],
    model_name: str,
) -> "pd.DataFrame":
    """
    Extract tokenizer-derived features using a pretrained tokenizer.

    Features:
      - len_chars: character length
      - char_entropy: Shannon entropy of character distribution
      - n_tokens: token count under this tokenizer
      - tok_rank_mean/min/max: token ID percentile statistics
      - zlib_bpc: zlib compression bits-per-character (baseline)
      - compression_ratio: n_tokens / len_chars (tokenizer efficiency)
      - max_token_len: length of longest token piece
      - mean_token_len: average token piece length
      - frac_single_char_tokens: fraction of tokens that are single characters
      - has_unk: whether any <unk> token appears
      - log_token_count: log(n_tokens)
    """
    import pandas as pd

    vocab_size = tokenizer.vocab_size or len(tokenizer)
    denom = max(vocab_size - 1, 1)

    rows = []
    for i, (cid, text) in enumerate(zip(candidate_ids, texts)):
        row: Dict[str, Any] = {
            "candidate_id": cid,
            "text_raw": text,
            "model_name": model_name,
        }

        if not text.strip():
            row.update({
                "len_chars": 0,
                "char_entropy": float("nan"),
                "n_tokens": 0,
                "tok_rank_mean": float("nan"),
                "tok_rank_min": float("nan"),
                "tok_rank_max": float("nan"),
                "zlib_bpc": float("nan"),
                "compression_ratio": float("nan"),
                "max_token_len": float("nan"),
                "mean_token_len": float("nan"),
                "frac_single_char_tokens": float("nan"),
                "has_unk": False,
                "log_token_count": float("nan"),
            })
            rows.append(row)
            continue

        # Basic features
        row["len_chars"] = len(text)
        row["char_entropy"] = _char_entropy(text)

        # Tokenize
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        n_tokens = len(token_ids)
        row["n_tokens"] = n_tokens

        # Token rank percentiles
        if n_tokens > 0:
            rank_pcts = [tid / denom for tid in token_ids]
            row["tok_rank_mean"] = float(np.mean(rank_pcts))
            row["tok_rank_min"] = float(np.min(rank_pcts))
            row["tok_rank_max"] = float(np.max(rank_pcts))
        else:
            row["tok_rank_mean"] = float("nan")
            row["tok_rank_min"] = float("nan")
            row["tok_rank_max"] = float("nan")

        # zlib baseline
        row["zlib_bpc"] = zlib_bpc(text)

        # Compression ratio
        row["compression_ratio"] = n_tokens / len(text) if len(text) > 0 else float("nan")

        # Token piece length stats
        if n_tokens > 0:
            pieces = tokenizer.convert_ids_to_tokens(token_ids)
            piece_lens = [len(p) for p in pieces]
            row["max_token_len"] = max(piece_lens)
            row["mean_token_len"] = float(np.mean(piece_lens))
            row["frac_single_char_tokens"] = sum(1 for pl in piece_lens if pl == 1) / n_tokens

            # Check for unknown tokens
            unk_id = tokenizer.unk_token_id
            row["has_unk"] = (unk_id is not None and unk_id in token_ids)
        else:
            row["max_token_len"] = float("nan")
            row["mean_token_len"] = float("nan")
            row["frac_single_char_tokens"] = float("nan")
            row["has_unk"] = False

        row["log_token_count"] = math.log(n_tokens) if n_tokens > 0 else float("nan")

        rows.append(row)

        if (i + 1) % 1000 == 0:
            logger.info("Extracted features for %d / %d candidates", i + 1, len(texts))

    return pd.DataFrame(rows)


def _char_entropy(s: str) -> float:
    """Shannon entropy over the character distribution (bits)."""
    if not s:
        return float("nan")
    freq: Dict[str, int] = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    n = len(s)
    return -sum((cnt / n) * math.log2(cnt / n) for cnt in freq.values())


# ---------------------------------------------------------------------------
# Memorization label construction for pretrained models
# ---------------------------------------------------------------------------

def build_pretrained_labels(
    target_scores: List[Dict],
    ref_scores: Optional[List[Dict]],
    candidate_ids: List[str],
    texts: List[str],
) -> "pd.DataFrame":
    """
    Construct memorization labels for pretrained model evaluation.

    Three complementary signals:
      1. bpc_target: absolute BPC (low = possibly memorized)
      2. delta_bpc: BPC(ref) - BPC(target)  (high = memorized by larger model)
      3. zlib_ratio: zlib_bpc / model_bpc  (high = model suspiciously good)

    Labels are assigned at multiple thresholds:
      - label_top_5pct, label_top_1pct, label_top_0_1pct (by delta_bpc or zlib_ratio)
    """
    import pandas as pd

    rows = []
    for i, cid in enumerate(candidate_ids):
        ts = target_scores[i]
        row = {
            "candidate_id": cid,
            "text_raw": texts[i],
            "bpc_target": ts["bpc"],
            "valid_target": ts["valid"],
        }

        # zlib ratio
        z = zlib_bpc(texts[i])
        row["zlib_bpc"] = z
        if ts["valid"] and ts["bpc"] > 0:
            row["zlib_ratio"] = z / ts["bpc"]
        else:
            row["zlib_ratio"] = float("nan")

        # Cross-scale delta
        if ref_scores is not None:
            rs = ref_scores[i]
            row["bpc_ref"] = rs["bpc"]
            row["valid_ref"] = rs["valid"]
            if ts["valid"] and rs["valid"]:
                row["delta_bpc"] = rs["bpc"] - ts["bpc"]
            else:
                row["delta_bpc"] = float("nan")
        else:
            row["bpc_ref"] = float("nan")
            row["valid_ref"] = False
            row["delta_bpc"] = float("nan")

        row["valid_label"] = ts["valid"] and (ref_scores is None or ref_scores[i]["valid"])
        rows.append(row)

    df = pd.DataFrame(rows)

    # Compute threshold labels on valid rows
    valid_mask = df["valid_label"]
    n_valid = valid_mask.sum()

    # Label by delta_bpc if available, else by zlib_ratio
    if ref_scores is not None:
        signal_col = "delta_bpc"
    else:
        signal_col = "zlib_ratio"

    for frac, col_name in [(0.05, "label_top_5pct"), (0.01, "label_top_1pct"), (0.001, "label_top_0_1pct")]:
        if col_name == "label_top_0_1pct" and n_valid < 1000:
            df[col_name] = float("nan")
            continue

        if n_valid > 0:
            vals = df.loc[valid_mask, signal_col]
            cutoff = vals.quantile(1.0 - frac)
            df[col_name] = False
            df.loc[valid_mask, col_name] = (df.loc[valid_mask, signal_col] >= cutoff)
        else:
            df[col_name] = float("nan")

    return df


# ---------------------------------------------------------------------------
# Full pretrained evaluation pipeline
# ---------------------------------------------------------------------------

def run_pretrained_eval(
    target_model_name: str,
    candidates_path: str,
    output_dir: str,
    run_id: str,
    ref_model_name: Optional[str] = None,
    batch_size: int = 32,
    max_length: int = 1024,
    device: str = "auto",
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run the complete pretrained model evaluation pipeline.

    Steps:
      1. Load candidates
      2. Load target model, extract features, score BPC
      3. (Optional) Load ref model, score BPC
      4. Build labels (cross-scale delta or zlib-ratio based)
      5. Train predictor on tokenizer features vs labels
      6. Evaluate with ablation

    Returns pipeline summary dict.
    """
    import pandas as pd

    logger.info("=" * 70)
    logger.info("PRETRAINED EVAL START  run_id=%s  target=%s  ref=%s",
                run_id, target_model_name, ref_model_name or "(none, zlib baseline)")
    logger.info("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    # --- Step 1: Load candidates ---
    from src.extract_features import load_candidates
    candidates = load_candidates(candidates_path)
    candidate_ids = [c["id"] for c in candidates]
    texts = [c["text"] for c in candidates]
    logger.info("Loaded %d candidates", len(candidates))

    # --- Step 2: Target model ---
    target_model, target_tokenizer, device = load_pretrained_model(target_model_name, device)

    # Features from target tokenizer
    logger.info("Extracting tokenizer features...")
    features_df = extract_pretrained_features(
        target_tokenizer, texts, candidate_ids, target_model_name,
    )
    features_path = os.path.join(output_dir, f"{run_id}_features.parquet")
    features_df.to_parquet(features_path, index=False)
    logger.info("Features saved: %s", features_path)

    # Score BPC with target model
    logger.info("Scoring BPC with target model...")
    target_scores = score_pretrained_bpc(
        target_model, target_tokenizer, texts, device,
        batch_size=batch_size, max_length=max_length,
    )

    # Free target model memory
    import torch
    del target_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Step 3: Reference model (optional) ---
    ref_scores = None
    if ref_model_name:
        ref_model, ref_tokenizer, device = load_pretrained_model(ref_model_name, device)
        logger.info("Scoring BPC with reference model...")
        ref_scores = score_pretrained_bpc(
            ref_model, ref_tokenizer, texts, device,
            batch_size=batch_size, max_length=max_length,
        )
        del ref_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Step 4: Build labels ---
    logger.info("Building memorization labels...")
    labels_df = build_pretrained_labels(target_scores, ref_scores, candidate_ids, texts)
    labels_path = os.path.join(output_dir, f"{run_id}_labels.parquet")
    labels_df.to_parquet(labels_path, index=False)
    logger.info("Labels saved: %s", labels_path)

    # --- Step 5: Train predictor with ablation ---
    logger.info("Training predictors with ablation...")
    ablation_results = run_ablation_study(
        features_df, labels_df, output_dir, run_id, seed=seed,
    )

    # --- Step 6: Save summary ---
    n_valid = int(labels_df["valid_label"].sum())
    summary = {
        "run_id": run_id,
        "target_model": target_model_name,
        "ref_model": ref_model_name,
        "n_candidates": len(candidates),
        "n_valid": n_valid,
        "device": device,
        "finished_at": datetime.datetime.utcnow().isoformat() + "Z",
        "ablation_results": ablation_results,
    }

    # Add label distribution stats
    if n_valid > 0:
        valid = labels_df[labels_df["valid_label"]]
        summary["stats"] = {
            "bpc_target_mean": float(valid["bpc_target"].mean()),
            "bpc_target_std": float(valid["bpc_target"].std()),
        }
        if "delta_bpc" in valid.columns:
            summary["stats"]["delta_bpc_mean"] = float(valid["delta_bpc"].mean())
            summary["stats"]["delta_bpc_std"] = float(valid["delta_bpc"].std())
        if "zlib_ratio" in valid.columns:
            summary["stats"]["zlib_ratio_mean"] = float(valid["zlib_ratio"].mean())
            summary["stats"]["zlib_ratio_std"] = float(valid["zlib_ratio"].std())

    manifest_path = os.path.join(output_dir, f"{run_id}_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 70)
    logger.info("PRETRAINED EVAL COMPLETE  manifest=%s", manifest_path)
    logger.info("=" * 70)

    return summary


# ---------------------------------------------------------------------------
# Ablation study — the key scientific contribution
# ---------------------------------------------------------------------------

def run_ablation_study(
    features_df: "pd.DataFrame",
    labels_df: "pd.DataFrame",
    output_dir: str,
    run_id: str,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Systematic ablation: train predictors with increasingly rich feature sets
    to measure the MARGINAL contribution of tokenizer-specific features.

    Feature groups (cumulative):
      1. trivial:   len_chars only
      2. baseline:  len_chars + char_entropy + zlib_bpc
      3. counts:    + n_tokens + compression_ratio + log_token_count
      4. full:      + tok_rank_* + max_token_len + mean_token_len + frac_single_char_tokens
    """
    import pandas as pd
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    # Merge features and labels
    merged = features_df.merge(labels_df, on="candidate_id", suffixes=("_feat", "_label"))
    valid = merged[merged["valid_label"] == True].copy()

    if len(valid) < 50:
        logger.warning("Only %d valid rows — skipping ablation", len(valid))
        return {"error": "too_few_valid_rows", "n_valid": len(valid)}

    # Define feature groups
    feature_groups = {
        "trivial": ["len_chars"],
        "baseline": ["len_chars", "char_entropy", "zlib_bpc"],
        "counts": ["len_chars", "char_entropy", "zlib_bpc",
                    "n_tokens", "compression_ratio", "log_token_count"],
        "full": ["len_chars", "char_entropy", "zlib_bpc",
                 "n_tokens", "compression_ratio", "log_token_count",
                 "tok_rank_mean", "tok_rank_min", "tok_rank_max",
                 "max_token_len", "mean_token_len", "frac_single_char_tokens"],
    }

    # Determine label columns
    label_cols = [c for c in ["label_top_5pct", "label_top_1pct", "label_top_0_1pct"]
                  if c in valid.columns and valid[c].notna().all()]

    # Regression target
    if "delta_bpc" in valid.columns:
        reg_target = "delta_bpc"
    elif "zlib_ratio" in valid.columns:
        reg_target = "zlib_ratio"
    else:
        reg_target = None

    # Train/test split
    np.random.seed(seed)
    train_idx, test_idx = train_test_split(
        range(len(valid)), test_size=0.2, random_state=seed,
    )

    results = []

    for group_name, feature_cols in feature_groups.items():
        # Filter to available columns
        available = [c for c in feature_cols if c in valid.columns]
        if not available:
            continue

        X = valid[available].values
        X_train, X_test = X[train_idx], X[test_idx]

        # Handle NaN
        scaler = StandardScaler()
        X_train_nan = np.nan_to_num(X_train, nan=0.0)
        X_test_nan = np.nan_to_num(X_test, nan=0.0)
        X_train_scaled = scaler.fit_transform(X_train_nan)
        X_test_scaled = scaler.transform(X_test_nan)

        # Classification
        for label_col in label_cols:
            y = valid[label_col].astype(int).values
            y_train, y_test = y[train_idx], y[test_idx]

            if y_test.sum() == 0 or y_test.sum() == len(y_test):
                continue

            # Logistic Regression
            try:
                clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
                clf.fit(X_train_scaled, y_train)
                y_prob = clf.predict_proba(X_test_scaled)[:, 1]
                auroc = roc_auc_score(y_test, y_prob)
                auprc = average_precision_score(y_test, y_prob)
                results.append({
                    "task": "classification",
                    "model": "logistic",
                    "feature_group": group_name,
                    "n_features": len(available),
                    "label": label_col,
                    "auroc": round(auroc, 6),
                    "auprc": round(auprc, 6),
                    "n_test": len(y_test),
                    "n_pos_test": int(y_test.sum()),
                })
            except Exception as e:
                logger.warning("Logistic failed for %s/%s: %s", group_name, label_col, e)

            # GBM (for full feature set only, to check nonlinear gains)
            if group_name == "full":
                try:
                    gbm = GradientBoostingClassifier(
                        n_estimators=200, max_depth=4, random_state=seed,
                    )
                    gbm.fit(X_train_scaled, y_train)
                    y_prob_gbm = gbm.predict_proba(X_test_scaled)[:, 1]
                    auroc_gbm = roc_auc_score(y_test, y_prob_gbm)
                    auprc_gbm = average_precision_score(y_test, y_prob_gbm)
                    results.append({
                        "task": "classification",
                        "model": "gbm",
                        "feature_group": group_name,
                        "n_features": len(available),
                        "label": label_col,
                        "auroc": round(auroc_gbm, 6),
                        "auprc": round(auprc_gbm, 6),
                        "n_test": len(y_test),
                        "n_pos_test": int(y_test.sum()),
                    })
                except Exception as e:
                    logger.warning("GBM failed for %s/%s: %s", group_name, label_col, e)

        # Regression
        if reg_target and reg_target in valid.columns:
            y_reg = valid[reg_target].values
            y_train_reg, y_test_reg = y_reg[train_idx], y_reg[test_idx]

            try:
                ridge = Ridge(alpha=1.0)
                ridge.fit(X_train_scaled, y_train_reg)
                y_pred = ridge.predict(X_test_scaled)
                from scipy.stats import pearsonr, spearmanr
                pearson_r = pearsonr(y_test_reg, y_pred)[0]
                spearman_r = spearmanr(y_test_reg, y_pred)[0]
                results.append({
                    "task": "regression",
                    "model": "ridge",
                    "feature_group": group_name,
                    "n_features": len(available),
                    "label": reg_target,
                    "pearson_r": round(float(pearson_r), 6),
                    "spearman_rho": round(float(spearman_r), 6),
                    "n_test": len(y_test_reg),
                })
            except Exception as e:
                logger.warning("Ridge failed for %s: %s", group_name, e)

    # Save ablation results
    results_df = pd.DataFrame(results)
    ablation_path = os.path.join(output_dir, f"{run_id}_ablation.parquet")
    results_df.to_parquet(ablation_path, index=False)
    logger.info("Ablation results saved: %s (%d rows)", ablation_path, len(results_df))

    # Print summary table
    if len(results_df) > 0:
        logger.info("\n=== ABLATION RESULTS ===")
        for _, row in results_df.iterrows():
            if row["task"] == "classification":
                logger.info(
                    "  %s | %-10s | %-8s | %-16s | AUROC=%.4f  AUPRC=%.4f",
                    row["task"], row["feature_group"], row["model"],
                    row["label"], row["auroc"], row["auprc"],
                )
            else:
                logger.info(
                    "  %s | %-10s | %-8s | %-16s | r=%.4f  rho=%.4f",
                    row["task"], row["feature_group"], row["model"],
                    row["label"], row["pearson_r"], row["spearman_rho"],
                )

    return {
        "n_results": len(results),
        "path": ablation_path,
        "groups_tested": list(feature_groups.keys()),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.pretrained_eval",
        description="Evaluate memorization risk prediction on pretrained LMs.",
    )
    p.add_argument("--model", required=True, help="HuggingFace model name (e.g. EleutherAI/pythia-1.4b)")
    p.add_argument("--ref-model", default=None, help="Smaller reference model for cross-scale delta")
    p.add_argument("--candidates", required=True, help="JSONL candidates file")
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument("--run-id", required=True, help="Run identifier")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    run_pretrained_eval(
        target_model_name=args.model,
        candidates_path=args.candidates,
        output_dir=args.output,
        run_id=args.run_id,
        ref_model_name=args.ref_model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
