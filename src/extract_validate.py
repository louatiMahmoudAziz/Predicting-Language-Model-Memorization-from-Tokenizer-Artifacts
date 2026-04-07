"""
extract_validate.py
-------------------
Validates memorization risk predictions via extraction experiments.

For each candidate string, attempts to extract it from the target model by:
  1. Prompting with the first K characters (prefix attack)
  2. Greedy decoding to generate the continuation
  3. Comparing the generated text against the actual candidate

Metrics:
  - Exact match rate: fraction of candidates where generation exactly matches
  - Longest common prefix (LCP): average/max matching prefix length
  - BLEU/ROUGE-L: approximate match quality
  - Extraction@K: among top-K predicted-risk candidates, what fraction extractable?

This provides a NON-CIRCULAR validation: predicted risk -> actual extractability.

Usage:
    python -m src.extract_validate \\
        --model-dir models/colab_max/target \\
        --tokenizer-meta tokenizers/max_bpe/metadata.json \\
        --predictions results/colab_max/predictor/predictions.parquet \\
        --candidates data/candidates/candidates_max.jsonl \\
        --output results/colab_max/extraction/ \\
        --run-id colab_max

    # For pretrained models:
    python -m src.extract_validate \\
        --pretrained EleutherAI/pythia-1.4b \\
        --predictions results/pythia_1b/pythia_1b_ablation.parquet \\
        --candidates data/candidates/pile_candidates.jsonl \\
        --output results/pythia_1b/extraction/ \\
        --run-id pythia_1b_extract
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Extraction via prefix prompting
# ---------------------------------------------------------------------------

def extract_by_prefix(
    model: Any,
    tokenizer: Any,
    text: str,
    device: str,
    prefix_frac: float = 0.5,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """
    Attempt to extract a candidate string by prompting with its prefix.

    Parameters
    ----------
    model       : loaded causal LM
    tokenizer   : corresponding tokenizer
    text        : the full candidate string
    prefix_frac : fraction of character length to use as prompt
    max_new_tokens : maximum tokens to generate
    temperature : 0.0 for greedy decoding

    Returns
    -------
    dict with keys:
        prefix, generated, target_suffix, exact_match, lcp_chars, lcp_ratio
    """
    import torch

    if not text.strip():
        return {
            "prefix": "",
            "generated": "",
            "target_suffix": "",
            "exact_match": False,
            "lcp_chars": 0,
            "lcp_ratio": 0.0,
        }

    # Split into prefix and target suffix
    split_point = max(1, int(len(text) * prefix_frac))
    prefix = text[:split_point]
    target_suffix = text[split_point:]

    if not target_suffix:
        return {
            "prefix": prefix,
            "generated": "",
            "target_suffix": "",
            "exact_match": True,
            "lcp_chars": 0,
            "lcp_ratio": 1.0,
        }

    # Encode prefix
    input_ids = tokenizer.encode(prefix, return_tensors="pt").to(device)

    # Generate continuation
    with torch.no_grad():
        if temperature == 0.0:
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        else:
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

    # Decode only the generated part
    generated_ids = output_ids[0, input_ids.shape[1]:]
    generated = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Compare
    exact_match = generated.startswith(target_suffix) or target_suffix.startswith(generated)

    # Longest common prefix
    lcp = 0
    for c1, c2 in zip(generated, target_suffix):
        if c1 == c2:
            lcp += 1
        else:
            break

    lcp_ratio = lcp / len(target_suffix) if target_suffix else 1.0

    return {
        "prefix": prefix,
        "generated": generated[:len(target_suffix) + 50],  # truncate for storage
        "target_suffix": target_suffix[:500],
        "exact_match": exact_match,
        "lcp_chars": lcp,
        "lcp_ratio": round(lcp_ratio, 6),
    }


# ---------------------------------------------------------------------------
# Batch extraction
# ---------------------------------------------------------------------------

def run_extraction_experiment(
    model: Any,
    tokenizer: Any,
    candidates: List[Dict],
    device: str,
    prefix_fracs: List[float] = [0.25, 0.5, 0.75],
    max_new_tokens: int = 256,
) -> "pd.DataFrame":
    """
    Run extraction experiments at multiple prefix fractions.

    Returns DataFrame with extraction results per candidate per prefix_frac.
    """
    import pandas as pd

    rows = []
    n = len(candidates)

    for i, cand in enumerate(candidates):
        cid = cand["id"]
        text = cand["text"]

        for frac in prefix_fracs:
            result = extract_by_prefix(
                model, tokenizer, text, device,
                prefix_frac=frac,
                max_new_tokens=max_new_tokens,
            )
            result["candidate_id"] = cid
            result["prefix_frac"] = frac
            rows.append(result)

        if (i + 1) % 100 == 0 or (i + 1) == n:
            logger.info("Extraction: %d / %d candidates", i + 1, n)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Extraction-vs-prediction correlation
# ---------------------------------------------------------------------------

def compute_extraction_metrics(
    extraction_df: "pd.DataFrame",
    predictions_df: "pd.DataFrame",
    k_values: List[int] = [50, 100, 200, 500],
) -> Dict[str, Any]:
    """
    Compute extraction validation metrics:
      - Overall extraction rate
      - Extraction@K: among top-K predicted-risk candidates, extraction rate
      - Correlation between predicted risk and extraction success
    """
    import pandas as pd
    from scipy.stats import spearmanr

    metrics = {}

    # Best extraction per candidate (across prefix fracs)
    best = extraction_df.groupby("candidate_id").agg({
        "exact_match": "max",
        "lcp_ratio": "max",
    }).reset_index()

    metrics["overall_exact_match_rate"] = float(best["exact_match"].mean())
    metrics["overall_mean_lcp_ratio"] = float(best["lcp_ratio"].mean())
    metrics["n_candidates"] = len(best)

    # Merge with predictions if available
    if predictions_df is not None and "candidate_id" in predictions_df.columns:
        # Find the score column (varies by pipeline)
        score_cols = [c for c in predictions_df.columns
                      if c.endswith("_prob") or c == "predicted_score" or c == "y_prob"]
        if not score_cols:
            # Try to use delta_bpc or zlib_ratio as proxy
            score_cols = [c for c in predictions_df.columns
                          if c in ("delta_bpc", "zlib_ratio", "bpc_target")]

        if score_cols:
            score_col = score_cols[0]
            merged = best.merge(predictions_df[["candidate_id", score_col]], on="candidate_id")

            if len(merged) > 10:
                # Correlation
                rho, p = spearmanr(merged[score_col], merged["lcp_ratio"])
                metrics["spearman_score_vs_lcp"] = round(float(rho), 6)
                metrics["spearman_p_value"] = round(float(p), 6)

                # Extraction@K
                sorted_by_risk = merged.sort_values(score_col, ascending=False)
                for k in k_values:
                    if k <= len(sorted_by_risk):
                        top_k = sorted_by_risk.head(k)
                        metrics[f"extraction_exact@{k}"] = float(top_k["exact_match"].mean())
                        metrics[f"extraction_lcp@{k}"] = float(top_k["lcp_ratio"].mean())

    return metrics


# ---------------------------------------------------------------------------
# Full validation pipeline
# ---------------------------------------------------------------------------

def validate_via_extraction(
    candidates_path: str,
    output_dir: str,
    run_id: str,
    model_dir: Optional[str] = None,
    tokenizer_meta_path: Optional[str] = None,
    pretrained_model_name: Optional[str] = None,
    predictions_path: Optional[str] = None,
    prefix_fracs: List[float] = [0.25, 0.5, 0.75],
    max_new_tokens: int = 256,
    max_candidates: int = 1000,
    device: str = "auto",
) -> Dict[str, Any]:
    """
    Run extraction experiments and validate predicted memorization risk.

    Supports both custom-trained models (model_dir + tokenizer_meta)
    and pretrained models (pretrained_model_name).
    """
    import pandas as pd
    import torch

    os.makedirs(output_dir, exist_ok=True)

    # Load candidates
    from src.extract_features import load_candidates
    candidates = load_candidates(candidates_path)
    if max_candidates and len(candidates) > max_candidates:
        logger.info("Limiting to %d candidates (from %d)", max_candidates, len(candidates))
        candidates = candidates[:max_candidates]

    # Load model
    if pretrained_model_name:
        from src.pretrained_eval import load_pretrained_model
        model, tokenizer, device = load_pretrained_model(pretrained_model_name, device)
    elif model_dir and tokenizer_meta_path:
        # Load custom-trained model
        from src.score_bpc import _load_model_and_tokenizer
        model, tokenizer, device = _load_model_and_tokenizer(
            model_dir, tokenizer_meta_path, device,
        )
    else:
        raise ValueError("Provide either --pretrained or --model-dir + --tokenizer-meta")

    # Run extraction
    logger.info("Running extraction experiments on %d candidates...", len(candidates))
    extraction_df = run_extraction_experiment(
        model, tokenizer, candidates, device,
        prefix_fracs=prefix_fracs,
        max_new_tokens=max_new_tokens,
    )

    # Save extraction results
    extraction_path = os.path.join(output_dir, f"{run_id}_extraction.parquet")
    extraction_df.to_parquet(extraction_path, index=False)
    logger.info("Extraction results saved: %s", extraction_path)

    # Load predictions if available
    predictions_df = None
    if predictions_path and os.path.isfile(predictions_path):
        predictions_df = pd.read_parquet(predictions_path)
        logger.info("Loaded predictions: %d rows", len(predictions_df))

    # Compute validation metrics
    metrics = compute_extraction_metrics(extraction_df, predictions_df)

    # Save metrics
    metrics_path = os.path.join(output_dir, f"{run_id}_extraction_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Extraction metrics saved: %s", metrics_path)

    # Log summary
    logger.info("=== EXTRACTION VALIDATION SUMMARY ===")
    logger.info("  Exact match rate: %.4f", metrics.get("overall_exact_match_rate", 0))
    logger.info("  Mean LCP ratio:   %.4f", metrics.get("overall_mean_lcp_ratio", 0))
    if "spearman_score_vs_lcp" in metrics:
        logger.info("  Spearman (risk vs LCP): %.4f (p=%.4f)",
                     metrics["spearman_score_vs_lcp"], metrics["spearman_p_value"])
    for k in [50, 100, 200, 500]:
        if f"extraction_exact@{k}" in metrics:
            logger.info("  Extraction@%d: exact=%.4f  lcp=%.4f",
                         k, metrics[f"extraction_exact@{k}"], metrics[f"extraction_lcp@{k}"])

    # Free model memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "run_id": run_id,
        "n_candidates": len(candidates),
        "extraction_path": extraction_path,
        "metrics": metrics,
        "finished_at": datetime.datetime.utcnow().isoformat() + "Z",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.extract_validate",
        description="Validate memorization predictions via extraction experiments.",
    )
    p.add_argument("--candidates", required=True, help="JSONL candidates file")
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument("--run-id", required=True, help="Run identifier")

    model_group = p.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--pretrained", help="HuggingFace pretrained model name")
    model_group.add_argument("--model-dir", help="Custom-trained model directory")

    p.add_argument("--tokenizer-meta", help="Tokenizer metadata path (for custom models)")
    p.add_argument("--predictions", help="Predictions parquet for correlation analysis")
    p.add_argument("--prefix-fracs", nargs="+", type=float, default=[0.25, 0.5, 0.75])
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--max-candidates", type=int, default=1000)
    p.add_argument("--device", default="auto")
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

    validate_via_extraction(
        candidates_path=args.candidates,
        output_dir=args.output,
        run_id=args.run_id,
        pretrained_model_name=args.pretrained,
        model_dir=args.model_dir,
        tokenizer_meta_path=args.tokenizer_meta,
        predictions_path=args.predictions,
        prefix_fracs=args.prefix_fracs,
        max_new_tokens=args.max_new_tokens,
        max_candidates=args.max_candidates,
        device=args.device,
    )


if __name__ == "__main__":
    main()
