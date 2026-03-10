"""
eval_metrics.py
---------------
Compute and report evaluation metrics for memorization risk predictors.

Pipeline position
-----------------
  train_predictor  ->  results/<run_id>/predictor/
                                                    |
                                                    v
  eval_metrics     ->  results/<run_id>/eval/

Inputs
------
  Predictor output directory containing:
    <run_id>_predictor_manifest.json         (top-level manifest listing all runs)
    <task>_<target>_<model>_<subset>/
      predictions.parquet                    (train/val/test predictions)
      metadata.json                          (feature list, threshold, counts)

Outputs
-------
  <output_dir>/
    <run_id>_eval_manifest.json              # top-level summary + all metrics
    comparison.parquet                       # flat table: one row per config
    comparison.json                          # same as comparison.parquet but JSON
    <task>_<target>_<model>_<subset>/
      metrics.json                           # full metric record for this config

Metric output schema (metrics.json)
------------------------------------
  {
    "task"          : "classification" | "regression",
    "target_col"    : str,
    "model_name"    : str,
    "feature_subset": str,
    "evaluated_at"  : ISO-8601 UTC string,
    "split"         : "test",
    "counts": {
      "n_test", "n_positive", "n_negative", "prevalence"
    },
    "ranking": {
      "score_column"    : "y_score" | "y_pred",
      "k_values"        : [50, 100, 500],
      "binarization"    : {...}  // regression only
      "precision_at_k"  : { "50": {value, resolvable, reason?}, ... },
      "recall_at_k"     : { "50": {...}, ... },
      "ndcg_at_k"       : { "50": {...}, ... }
    },
    "tail": {
      "tpr_at_fpr_1pct"  : {value, resolvable, reason?, fpr_target, fpr_achieved,
                             n_fp_at_threshold, n_negative, interpolated},
      "tpr_at_fpr_0_1pct": {value, resolvable, reason?, ...}
    },
    "secondary": {
      "auroc": {value, resolvable, reason?},
      "auprc": {value, resolvable, reason?}
    },
    "regression_metrics": null | {mse, rmse, pearson_r, spearman_rho, resolvable}
  }

Threshold / interpolation policy
---------------------------------
  TPR@FPR is computed using a step-function ROC curve (sklearn roc_curve).
  No linear interpolation is applied between operating points.
  At each unique score threshold, the ROC curve has one step.
  We select the last operating point where FPR <= target_fpr and report that TPR.
  This is conservative and fully auditable.
  The actual achieved FPR is always recorded alongside the target.

  Rationale for no interpolation: interpolation introduces bias for small test
  sets and is not standard in privacy-risk reporting. The step-function result
  is reproducible and conservative (it never overstates TPR).

Conditions under which a metric is marked unresolvable or skipped
------------------------------------------------------------------
  precision@K : K > n_test
  recall@K    : K > n_test  |  n_positive == 0
  NDCG@K      : K > n_test  |  n_positive == 0  |  IDCG == 0 (never if n_pos > 0)
  AUROC       : n_positive == 0  |  n_negative == 0
  AUPRC       : n_positive == 0  |  n_negative == 0
  TPR@1%FPR   : n_positive == 0  |  n_negative == 0  |  no ROC point at target
  TPR@0.1%FPR : same as above + n_negative < fpr_0_1_min_negatives (SPEC §7.3)
  regression  : n_test < 2  (Pearson/Spearman undefined)

  Note: precision@K with n_positive == 0 is reported as 0.0 (well-defined).

Regression ranking metrics
--------------------------
  For regression, ranking/tail metrics require binary ground truth.
  y_true (delta_bpc) is binarized at the top-(1 - reg_binarize_quantile) fraction
  of the test set (default: top 5%, quantile=0.95).
  This binarization is:
    - computed only from test-set y_true (no val/train data used)
    - recorded explicitly in the ranking.binarization sub-object
    - separate from classification label columns

CLI
---
  python -m src.eval_metrics \\
      --predictor-dir results/run1/predictor \\
      --run-id run1 \\
      --output results/run1/eval

  python -m src.eval_metrics --config configs/colab_mini.yaml
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FPR_1_PCT   = 0.01    # 1.0% FPR — always reported
_FPR_0_1_PCT = 0.001   # 0.1% FPR — subject to SPEC resolvability rule
_DEFAULT_REG_BINARIZE_QUANTILE = 0.95   # top 5% of test y_true as positives


# ---------------------------------------------------------------------------
# Metric building blocks
# ---------------------------------------------------------------------------

def _unresolvable(reason: str) -> Dict[str, Any]:
    """Return a standard 'not resolvable' metric record."""
    return {"value": None, "resolvable": False, "reason": reason}


def _resolved(value: float, **extra: Any) -> Dict[str, Any]:
    """Return a standard resolved metric record with optional extra fields."""
    return {"value": round(float(value), 8), "resolvable": True, **extra}


def _precision_at_k(
    y_true: np.ndarray, y_score: np.ndarray, k: int
) -> Dict[str, Any]:
    """
    Precision@K: fraction of top-K (by score) that are true positives.

    Score ties are broken by the original array order (argsort is stable).
    With n_positive == 0: returns 0.0 (well-defined; no positives to retrieve).
    """
    n = len(y_true)
    if n < 1:
        return _unresolvable("empty_test_set")
    if k > n:
        return _unresolvable(f"K={k}_gt_n_test={n}")
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return _resolved(0.0, n_positive=0, note="no_positives_precision_is_0")
    sort_idx = np.argsort(y_score, kind="stable")[::-1]
    top_k_labels = y_true[sort_idx[:k]]
    return _resolved(float(top_k_labels.sum() / k), n_positive=n_pos)


def _recall_at_k(
    y_true: np.ndarray, y_score: np.ndarray, k: int
) -> Dict[str, Any]:
    """
    Recall@K: fraction of all true positives that appear in the top-K by score.

    With n_positive == 0: unresolvable (denominator would be 0).
    """
    n = len(y_true)
    if n < 1:
        return _unresolvable("empty_test_set")
    if k > n:
        return _unresolvable(f"K={k}_gt_n_test={n}")
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return _unresolvable("no_positives_in_test")
    sort_idx = np.argsort(y_score, kind="stable")[::-1]
    top_k_labels = y_true[sort_idx[:k]]
    return _resolved(float(top_k_labels.sum() / n_pos), n_positive=n_pos)


def _ndcg_at_k(
    y_true: np.ndarray, y_score: np.ndarray, k: int
) -> Dict[str, Any]:
    """
    NDCG@K: Normalised Discounted Cumulative Gain at K.

    For binary relevance (rel ∈ {0, 1}):
      DCG@K  = Σ_{i=1}^{K}  rel_i / log2(i + 1)
      IDCG@K = Σ_{i=1}^{min(K, n_pos)} 1 / log2(i + 1)
      NDCG@K = DCG@K / IDCG@K

    Score ties are broken by the original array order (stable argsort).
    """
    n = len(y_true)
    if n < 1:
        return _unresolvable("empty_test_set")
    if k > n:
        return _unresolvable(f"K={k}_gt_n_test={n}")
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return _unresolvable("no_positives_in_test")

    sort_idx = np.argsort(y_score, kind="stable")[::-1]
    top_k = y_true[sort_idx[:k]].astype(float)

    positions = np.arange(1, k + 1, dtype=float)     # ranks 1 .. K
    discounts = np.log2(positions + 1.0)              # log2(2), log2(3), ...
    dcg = float(np.sum(top_k / discounts))

    # Ideal: place all n_pos positives first (up to K)
    n_ideal = min(k, n_pos)
    ideal_positions = np.arange(1, n_ideal + 1, dtype=float)
    idcg = float(np.sum(1.0 / np.log2(ideal_positions + 1.0)))

    if idcg == 0.0:
        # Should never happen when n_pos > 0, but guard anyway
        return _unresolvable("idcg_zero_despite_positive_labels_bug")

    return _resolved(dcg / idcg, n_positive=n_pos)


def _tpr_at_fpr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_fpr: float,
    fpr_0_1_min_negatives: int,
) -> Dict[str, Any]:
    """
    TPR at the operating point where FPR <= target_fpr.

    Method: step-function ROC curve (sklearn roc_curve), no interpolation.
    Selects the last point on the curve where FPR <= target_fpr; reports that
    TPR and the actual achieved FPR (which may be strictly less than target_fpr).

    For target_fpr == 0.001 (0.1%): enforces SPEC §7.3 resolvability rule
    — n_negative must be >= fpr_0_1_min_negatives.

    Parameters
    ----------
    y_true                : binary ground truth (0/1)
    y_score               : predicted scores (higher = more likely positive)
    target_fpr            : e.g. 0.01 or 0.001
    fpr_0_1_min_negatives : minimum negatives required for 0.1% FPR (SPEC rule)
    """
    from sklearn.metrics import roc_curve  # type: ignore[import-untyped]

    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)

    if n_pos == 0:
        return _unresolvable("no_positives_in_test")
    if n_neg == 0:
        return _unresolvable("no_negatives_in_test")

    # SPEC §7.3: 0.1% FPR is only meaningful with enough negatives
    if abs(target_fpr - _FPR_0_1_PCT) < 1e-10:
        if n_neg < fpr_0_1_min_negatives:
            return _unresolvable(
                f"n_negative={n_neg}_lt_fpr_0_1_min_negatives={fpr_0_1_min_negatives}"
            )

    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_score)

    # Find last index where FPR <= target (step-function — no interpolation)
    eligible_mask = fpr_arr <= target_fpr
    if not eligible_mask.any():
        # roc_curve always starts at (0, 0), so this should not happen
        return _unresolvable("no_roc_point_at_or_below_target_fpr")

    last_idx = int(np.where(eligible_mask)[0][-1])
    fpr_achieved = float(fpr_arr[last_idx])
    tpr_achieved = float(tpr_arr[last_idx])
    n_fp_at_threshold = int(round(fpr_achieved * n_neg))

    # Detect when FPR resolution is too coarse for the target.
    # If n_neg < ceil(1/target_fpr), the smallest non-zero FPR step (1/n_neg)
    # exceeds the target, so the metric is constrained to the trivial (0, 0)
    # or only to operating points where no false positives exist. The result
    # is technically correct but may be misleading in comparison tables.
    min_achievable_fpr = 1.0 / n_neg if n_neg > 0 else float("inf")
    low_resolution = min_achievable_fpr > target_fpr

    if low_resolution:
        logger.debug(
            "TPR@FPR<=%.4f: low resolution (n_neg=%d, min_fpr_step=%.4f > target=%.4f). "
            "Metric constrained to FPR=0 operating points.",
            target_fpr, n_neg, min_achievable_fpr, target_fpr,
        )

    return _resolved(
        tpr_achieved,
        fpr_target=float(target_fpr),
        fpr_achieved=fpr_achieved,
        n_fp_at_threshold=n_fp_at_threshold,
        n_negative=n_neg,
        interpolated=False,
        low_fpr_resolution=low_resolution,
        min_achievable_fpr=round(min_achievable_fpr, 10),
    )


def _auroc(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, Any]:
    """AUROC (secondary metric per SPEC §7.2)."""
    from sklearn.metrics import roc_auc_score  # type: ignore[import-untyped]

    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0:
        return _unresolvable("no_positives_in_test")
    if n_neg == 0:
        return _unresolvable("no_negatives_in_test")
    return _resolved(float(roc_auc_score(y_true, y_score)))


def _auprc(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, Any]:
    """AUPRC (secondary metric per SPEC §7.2)."""
    from sklearn.metrics import average_precision_score  # type: ignore[import-untyped]

    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0:
        return _unresolvable("no_positives_in_test")
    if n_neg == 0:
        return _unresolvable("no_negatives_in_test")
    return _resolved(float(average_precision_score(y_true, y_score)))


def _regression_metrics_dict(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, Any]:
    """MSE, RMSE, Pearson r, Spearman rho for regression outputs."""
    from scipy.stats import pearsonr, spearmanr  # type: ignore[import-untyped]

    n = len(y_true)
    mse  = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))

    if n < 2:
        return {
            "mse": mse, "rmse": rmse,
            "pearson_r": None, "spearman_rho": None,
            "resolvable": False, "reason": "n_test_lt_2",
        }
    pr, _ = pearsonr(y_true, y_pred)
    sr, _ = spearmanr(y_true, y_pred)
    return {
        "mse": mse, "rmse": rmse,
        "pearson_r": float(pr), "spearman_rho": float(sr),
        "resolvable": True,
    }


def _binarize_for_ranking(
    y_true_continuous: np.ndarray,
    quantile: float,
) -> Tuple[np.ndarray, float, int]:
    """
    Binarize continuous y_true at the given quantile for regression ranking.

    Returns (binary_labels, threshold_value, n_positives).
    Rows >= threshold are labelled positive (1).

    Example: quantile=0.95 -> top 5% of test-set delta_bpc as positives.
    """
    threshold = float(np.quantile(y_true_continuous, quantile))
    binary = (y_true_continuous >= threshold).astype(float)
    n_pos = int(binary.sum())
    return binary, threshold, n_pos


# ---------------------------------------------------------------------------
# Per-config evaluator
# ---------------------------------------------------------------------------

def _eval_one_config(
    task: str,
    target_col: str,
    model_name: str,
    subset_name: str,
    predictions_df: Any,
    fpr_0_1_min_negatives: int,
    ranking_k: List[int],
    reg_binarize_quantile: float,
) -> Dict[str, Any]:
    """
    Compute all metrics for one (task, model, feature_subset) on its test split.

    All scoring is done on the test split only. The val split is used by
    train_predictor.py for threshold selection; it must not be used here.

    For classification: y_score = predicted probability; y_true = 0/1 label.
    For regression:     y_score = y_pred (predicted delta_bpc);
                        y_true  = actual delta_bpc, binarized for ranking metrics.
    """
    df_test = predictions_df[predictions_df["split"] == "test"].copy()
    n_test = len(df_test)

    common = {
        "task": task,
        "target_col": target_col,
        "model_name": model_name,
        "feature_subset": subset_name,
        "evaluated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "split": "test",
    }

    if n_test == 0:
        return {**common, "error": "no_test_split_rows_in_predictions_parquet"}

    y_true_raw = df_test["y_true"].values.astype(float)
    is_clf = task == "classification"

    # ---- NaN guard (CRITICAL: NaN in scores silently corrupts ranking) -----
    n_nan_y_true = int(np.isnan(y_true_raw).sum())
    if n_nan_y_true > 0:
        return {
            **common,
            "error": (
                f"y_true contains {n_nan_y_true} NaN values in test split. "
                "This is an upstream invariant violation (train_predictor / build_labels)."
            ),
        }

    # ---- Resolve score column and binary ground truth ----------------------
    if is_clf:
        if "y_score" not in df_test.columns:
            return {**common, "error": "predictions_parquet_missing_y_score_column"}
        y_score = df_test["y_score"].values.astype(float)
    else:  # regression
        if "y_pred" not in df_test.columns:
            return {**common, "error": "predictions_parquet_missing_y_pred_column"}
        y_score = df_test["y_pred"].values.astype(float)

    # Drop rows where y_score is NaN — argsort sends NaN to position #1
    # in descending order, silently corrupting all ranking metrics.
    nan_score_mask = np.isnan(y_score)
    n_nan_score = int(nan_score_mask.sum())
    if n_nan_score > 0:
        logger.warning(
            "%s/%s/%s: dropping %d rows with NaN predictions (%d -> %d test rows)",
            task, model_name, subset_name, n_nan_score, n_test, n_test - n_nan_score,
        )
        keep = ~nan_score_mask
        y_true_raw = y_true_raw[keep]
        y_score = y_score[keep]
        n_test = len(y_score)
        if n_test == 0:
            return {**common, "error": "all_test_predictions_are_NaN"}

    common["n_nan_scores_dropped"] = n_nan_score

    if is_clf:
        # y_true is already 0.0/1.0 from train_predictor
        y_binary = (y_true_raw >= 0.5).astype(float)
        binarization_info: Optional[Dict[str, Any]] = None
    else:  # regression
        # Binarize y_true (delta_bpc) at the top-(1-quantile) fraction of test set
        y_binary, bin_thr, bin_n_pos = _binarize_for_ranking(
            y_true_raw, reg_binarize_quantile
        )
        pct_str = f"top_{int(round((1.0 - reg_binarize_quantile) * 100))}pct"
        binarization_info = {
            "method": f"{pct_str}_of_test_y_true",
            "quantile": float(reg_binarize_quantile),
            "threshold_value": float(bin_thr),
            "n_positives": bin_n_pos,
            "note": (
                "Binary labels derived from test-set y_true quantile only. "
                "Different from build_labels.py thresholds."
            ),
        }

    n_pos = int(y_binary.sum())
    n_neg = n_test - n_pos

    # ---- Ranking metrics ---------------------------------------------------
    prec_at_k: Dict[str, Any] = {}
    rec_at_k:  Dict[str, Any] = {}
    ndcg_at_k: Dict[str, Any] = {}

    for k in ranking_k:
        prec_at_k[str(k)] = _precision_at_k(y_binary, y_score, k)
        rec_at_k[str(k)]  = _recall_at_k(y_binary, y_score, k)
        ndcg_at_k[str(k)] = _ndcg_at_k(y_binary, y_score, k)

    ranking: Dict[str, Any] = {
        "score_column": "y_score" if is_clf else "y_pred",
        "k_values": ranking_k,
        "precision_at_k": prec_at_k,
        "recall_at_k":    rec_at_k,
        "ndcg_at_k":      ndcg_at_k,
    }
    if binarization_info is not None:
        ranking["binarization"] = binarization_info

    # ---- Tail metrics ------------------------------------------------------
    tail: Dict[str, Any] = {
        "tpr_at_fpr_1pct": _tpr_at_fpr(
            y_binary, y_score, _FPR_1_PCT, fpr_0_1_min_negatives
        ),
        "tpr_at_fpr_0_1pct": _tpr_at_fpr(
            y_binary, y_score, _FPR_0_1_PCT, fpr_0_1_min_negatives
        ),
    }

    # ---- Secondary metrics -------------------------------------------------
    secondary: Dict[str, Any] = {
        "auroc": _auroc(y_binary, y_score),
        "auprc": _auprc(y_binary, y_score),
    }

    # ---- Regression-only metrics ------------------------------------------
    regression_metrics: Optional[Dict[str, Any]] = None
    if not is_clf:
        regression_metrics = _regression_metrics_dict(y_true_raw, y_score)

    return {
        **common,
        "counts": {
            "n_test":     n_test,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "prevalence": float(n_pos / n_test) if n_test > 0 else None,
        },
        "ranking":            ranking,
        "tail":               tail,
        "secondary":          secondary,
        "regression_metrics": regression_metrics,
    }


# ---------------------------------------------------------------------------
# Comparison-table flattener
# ---------------------------------------------------------------------------

def _flatten_for_comparison(
    metrics: Dict[str, Any],
    run_meta: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Collapse a full metrics dict to one flat row for the comparison table.

    Unresolvable metrics are recorded as None (null in JSON/NaN in parquet).
    """
    row: Dict[str, Any] = {
        "task":           metrics.get("task"),
        "target_col":     metrics.get("target_col"),
        "model_name":     metrics.get("model_name"),
        "feature_subset": metrics.get("feature_subset"),
        "error":          metrics.get("error"),
    }

    if "error" in metrics:
        return row

    counts = metrics.get("counts", {})
    row.update({
        "n_test":     counts.get("n_test"),
        "n_positive": counts.get("n_positive"),
        "n_negative": counts.get("n_negative"),
        "prevalence": counts.get("prevalence"),
    })

    # Ranking — precision, recall, NDCG
    ranking = metrics.get("ranking", {})
    for metric_key, k_results in [
        ("precision_at_k", ranking.get("precision_at_k", {})),
        ("recall_at_k",    ranking.get("recall_at_k",    {})),
        ("ndcg_at_k",      ranking.get("ndcg_at_k",      {})),
    ]:
        for k_str, result in k_results.items():
            col = f"{metric_key}_{k_str}"
            row[col] = result.get("value") if result.get("resolvable") else None

    # Tail metrics (preserve low_fpr_resolution flag for interpretability)
    tail = metrics.get("tail", {})
    for metric_name, result in tail.items():
        row[metric_name] = result.get("value") if result.get("resolvable") else None
        if result.get("low_fpr_resolution"):
            row[f"{metric_name}_low_resolution"] = True

    # Secondary
    secondary = metrics.get("secondary", {})
    for metric_name, result in secondary.items():
        row[metric_name] = result.get("value") if result.get("resolvable") else None

    # Regression
    reg = metrics.get("regression_metrics")
    if reg:
        for key in ("mse", "rmse", "pearson_r", "spearman_rho"):
            row[f"reg_{key}"] = reg.get(key)

    row["n_features_used"] = run_meta.get("n_features_used")
    return row


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

def eval_metrics(
    predictor_dir: str,
    run_id: str,
    output_dir: str,
    *,
    fpr_0_1_min_negatives: int = 100000,
    ranking_k: Optional[List[int]] = None,
    reg_binarize_quantile: float = _DEFAULT_REG_BINARIZE_QUANTILE,
) -> str:
    """
    Evaluate all predictor runs found in predictor_dir.

    Reads <run_id>_predictor_manifest.json and each run's predictions.parquet.
    Computes metrics on the test split only. Saves per-run metrics.json files,
    a flat comparison table, and a top-level eval manifest.

    Parameters
    ----------
    predictor_dir         : directory containing the predictor manifest + run subdirs
    run_id                : experiment identifier (must match predictor manifest)
    output_dir            : directory for eval outputs
    fpr_0_1_min_negatives : SPEC §7.3 minimum negatives for 0.1% FPR
    ranking_k             : K values for precision/recall/NDCG (default: [50, 100, 500])
    reg_binarize_quantile : quantile for binarizing regression y_true for ranking
                            (default 0.95 = top 5% as positives)

    Returns
    -------
    str  Absolute path to the top-level eval manifest JSON.
    """
    import pandas as pd  # type: ignore[import-untyped]

    if ranking_k is None:
        ranking_k = [50, 100, 500]

    logger.info("=" * 65)
    logger.info("eval_metrics: run_id=%s", run_id)
    logger.info("=" * 65)

    # ------------------------------------------------------------------
    # Load predictor manifest
    # ------------------------------------------------------------------
    manifest_path = os.path.join(predictor_dir, f"{run_id}_predictor_manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f"Predictor manifest not found: {manifest_path!r}. "
            f"Run train_predictor.py first."
        )

    with open(manifest_path, "r", encoding="utf-8") as fh:
        pred_manifest = json.load(fh)

    runs = pred_manifest.get("runs", [])
    logger.info("Predictor manifest: %d runs listed", len(runs))

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Evaluate each run
    # ------------------------------------------------------------------
    all_metrics: List[Dict[str, Any]] = []
    comparison_rows: List[Dict[str, Any]] = []
    n_skipped = 0

    for run_entry in runs:
        task         = run_entry.get("task",           "?")
        target_col   = run_entry.get("target_col",     "?")
        model_name   = run_entry.get("model_name",     "?")
        subset_name  = run_entry.get("feature_subset", "?")
        run_dir      = run_entry.get("run_dir",        "")

        # Skip runs that failed during training
        if "error" in run_entry:
            logger.warning(
                "Skipping run %s/%s/%s: training error: %s",
                task, model_name, subset_name, run_entry["error"],
            )
            n_skipped += 1
            continue

        pred_path = os.path.join(run_dir, "predictions.parquet")
        meta_path = os.path.join(run_dir, "metadata.json")

        if not os.path.isfile(pred_path):
            logger.error(
                "predictions.parquet missing for %s/%s/%s — skipping: %s",
                task, model_name, subset_name, pred_path,
            )
            n_skipped += 1
            continue

        # Load predictions
        try:
            predictions_df = pd.read_parquet(pred_path)
        except Exception as exc:
            logger.error(
                "Failed to load predictions for %s/%s/%s: %s",
                task, model_name, subset_name, exc,
            )
            n_skipped += 1
            continue

        # Load per-run metadata (optional — used for comparison table)
        run_meta: Dict[str, Any] = {}
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as fh:
                    run_meta = json.load(fh)
            except Exception as exc:
                logger.warning("Could not load metadata.json for %s: %s", run_dir, exc)

        # Compute metrics
        try:
            metrics = _eval_one_config(
                task=task,
                target_col=target_col,
                model_name=model_name,
                subset_name=subset_name,
                predictions_df=predictions_df,
                fpr_0_1_min_negatives=fpr_0_1_min_negatives,
                ranking_k=ranking_k,
                reg_binarize_quantile=reg_binarize_quantile,
            )
        except Exception as exc:
            logger.error(
                "Metric computation failed for %s/%s/%s: %s",
                task, model_name, subset_name, exc,
            )
            metrics = {
                "task": task, "target_col": target_col,
                "model_name": model_name, "feature_subset": subset_name,
                "error": str(exc),
            }

        # Save per-config metrics.json
        target_short = target_col.replace("label_", "")
        dir_name = f"{task}_{target_short}_{model_name}_{subset_name}"
        config_eval_dir = os.path.join(output_dir, dir_name)
        os.makedirs(config_eval_dir, exist_ok=True)

        metrics_path = os.path.join(config_eval_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)

        all_metrics.append(metrics)
        comparison_rows.append(_flatten_for_comparison(metrics, run_meta))

        if "error" not in metrics:
            counts = metrics.get("counts", {})
            auroc_entry = metrics.get("secondary", {}).get("auroc", {})
            auroc_str = (
                f"AUROC={auroc_entry['value']:.4f}"
                if auroc_entry.get("resolvable") else "AUROC=n/a"
            )
            logger.info(
                "  [%s | %s | %s]  n_test=%d  n_pos=%d  %s",
                task, model_name, subset_name,
                counts.get("n_test", 0), counts.get("n_positive", 0), auroc_str,
            )
        else:
            logger.warning(
                "  [%s | %s | %s]  ERROR: %s",
                task, model_name, subset_name, metrics["error"],
            )

    # ------------------------------------------------------------------
    # Save comparison table
    # ------------------------------------------------------------------
    if comparison_rows:
        df_comparison = pd.DataFrame(comparison_rows)
        # Sort for readability: task, target_col, model_name, feature_subset
        sort_cols = [c for c in ["task", "target_col", "model_name", "feature_subset"]
                     if c in df_comparison.columns]
        if sort_cols:
            df_comparison = df_comparison.sort_values(sort_cols).reset_index(drop=True)

        comp_parquet = os.path.join(output_dir, "comparison.parquet")
        comp_json    = os.path.join(output_dir, "comparison.json")
        df_comparison.to_parquet(comp_parquet, index=False, engine="pyarrow")
        df_comparison.to_json(comp_json, orient="records", indent=2)
        logger.info("Comparison table -> %s  (%d rows)", comp_parquet, len(df_comparison))
    else:
        logger.warning("No successfully evaluated runs — comparison table not written.")

    # ------------------------------------------------------------------
    # Write top-level eval manifest
    # ------------------------------------------------------------------
    n_ok   = sum(1 for m in all_metrics if "error" not in m)
    n_err  = len(all_metrics) - n_ok

    eval_manifest = {
        "run_id":       run_id,
        "evaluated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "predictor_manifest": os.path.abspath(manifest_path),
        "n_runs_evaluated": len(all_metrics),
        "n_runs_ok":        n_ok,
        "n_runs_failed":    n_err,
        "n_runs_skipped":   n_skipped,
        "config": {
            "ranking_k":              ranking_k,
            "fpr_0_1_min_negatives":  fpr_0_1_min_negatives,
            "reg_binarize_quantile":  reg_binarize_quantile,
            "tpr_fpr_interpolation":  "step_function_no_interpolation",
        },
        "output_dir":  os.path.abspath(output_dir),
        "runs":        all_metrics,
    }

    manifest_out = os.path.join(output_dir, f"{run_id}_eval_manifest.json")
    with open(manifest_out, "w", encoding="utf-8") as fh:
        json.dump(eval_manifest, fh, indent=2)

    logger.info(
        "Completed: %d ok / %d failed / %d skipped -> %s",
        n_ok, n_err, n_skipped, manifest_out,
    )
    logger.info("=" * 65)
    return os.path.abspath(manifest_out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.eval_metrics",
        description="Evaluate memorization risk predictor outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",        metavar="YAML", help="Project YAML config.")
    p.add_argument("--predictor-dir", metavar="DIR",  help="Predictor output directory.")
    p.add_argument("--output",        metavar="DIR",  help="Eval output directory.")
    p.add_argument("--run-id",        metavar="ID")
    p.add_argument(
        "--fpr-0-1-min-negatives", type=int,
        help="Minimum negatives for 0.1%% FPR (SPEC §7.3).",
    )
    p.add_argument(
        "--ranking-k", type=int, nargs="+", metavar="K",
        help="K values for precision/recall/NDCG@K. Example: --ranking-k 50 100 500",
    )
    p.add_argument(
        "--reg-binarize-quantile", type=float,
        help="Quantile for binarizing regression y_true for ranking metrics. "
             "Default 0.95 (top 5%%).",
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

    predictor_dir = output_dir = run_id = None
    fpr_0_1_min_negatives = 100000
    ranking_k: Optional[List[int]] = None
    reg_binarize_quantile: float = _DEFAULT_REG_BINARIZE_QUANTILE

    if args.config:
        from src.config import load_config  # type: ignore[import]
        cfg = load_config(args.config)
        run_id = cfg.run_id
        fpr_0_1_min_negatives = cfg.evaluation.fpr_0_1_min_negatives
        ranking_k = cfg.evaluation.ranking_k
        predictor_dir = os.path.join(cfg.paths.results_dir, run_id, "predictor")
        output_dir    = os.path.join(cfg.paths.results_dir, run_id, "eval")

    if args.predictor_dir:        predictor_dir = args.predictor_dir
    if args.output:               output_dir    = args.output
    if args.run_id:               run_id        = args.run_id
    if args.fpr_0_1_min_negatives is not None:
        fpr_0_1_min_negatives = args.fpr_0_1_min_negatives
    if args.ranking_k:            ranking_k     = args.ranking_k
    if args.reg_binarize_quantile is not None:
        reg_binarize_quantile = args.reg_binarize_quantile

    missing = [
        name for name, val in [
            ("--predictor-dir", predictor_dir),
            ("--output",        output_dir),
            ("--run-id",        run_id),
        ]
        if val is None
    ]
    if missing:
        parser.error("Missing required arguments:\n  " + "\n  ".join(missing))

    try:
        out = eval_metrics(
            predictor_dir=predictor_dir,        # type: ignore[arg-type]
            run_id=run_id,                      # type: ignore[arg-type]
            output_dir=output_dir,              # type: ignore[arg-type]
            fpr_0_1_min_negatives=fpr_0_1_min_negatives,
            ranking_k=ranking_k,
            reg_binarize_quantile=reg_binarize_quantile,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("FATAL: %s", exc)
        sys.exit(1)

    print(f"\nDone.  Eval manifest written to: {out}")


if __name__ == "__main__":
    main()
