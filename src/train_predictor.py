"""
train_predictor.py
------------------
Train memorization risk predictors over tokenizer-only features and ΔBPC labels.

Pipeline position:
  extract_features  ──► features/<run_id>.parquet ──┐
                                                     ├──► train_predictor
  build_labels      ──► labels/<run_id>_labels.parquet ─┘

Tasks (both run in a single call)
----------------------------------
  regression      : predict delta_bpc (continuous)
  classification  : predict label_top_5pct / label_top_1pct / label_top_0_1pct

Train/val/test split policy
----------------------------
  - Computed ONCE over all valid_label=True rows.
  - Seeded numpy permutation -> fully deterministic.
  - train and val sizes are floor-rounded; test gets the remainder.
  - Split recorded in <run_id>_split.json by candidate_id.
  - Same split used for all (task, feature_subset, model) combinations.
  - No stratification; the SPEC requires only "fixed split with seed".

Feature subsets
---------------
  token_count_only      : n_tokens_target
  len_entropy           : len_chars, char_entropy
  token_count_delta_tok : n_tokens_target + all ref_*_delta_tok columns
  full                  : all numeric columns (excluding identity, labels, metadata)

NaN handling
------------
  1. Filter to valid_label=True before anything else.
  2. Assert no NaN in delta_bpc for filtered rows (upstream invariant).
  3. For classification: skip any label column that is all-NaN after filtering
     (signals that build_labels.py marked it not resolvable).
  4. For features: drop columns that are entirely NaN in training split.
     Impute remaining NaN with training-split median.
     Apply same imputer (same medians) to val and test — no leakage.
  5. Scaling (StandardScaler): applied only for linear models (logistic, ridge).

Model support
-------------
  Classification : logistic | xgboost | rf   (configured via model_type)
  Regression     : ridge (always, as explicit baseline) + configured model

Output layout
-------------
  <output_dir>/
    <run_id>_split.json                          # train/val/test candidate_ids
    regression_delta_bpc_ridge_<subset>/         # one dir per (task, model, subset)
      model.pkl
      predictions.parquet
      val_metrics.json
      test_metrics.json
      metadata.json
    classification_<label>_<model>_<subset>/
      ...
    <run_id>_predictor_manifest.json             # summary of all runs

CLI
---
  python -m src.train_predictor \\
      --features features/run1.parquet \\
      --labels   labels/run1_labels.parquet \\
      --output   results/run1 --run-id run1

  python -m src.train_predictor --config configs/colab_mini.yaml
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
# Feature column definitions (from extract_features.py schema)
# ---------------------------------------------------------------------------

_BASELINE_COLS = ["len_chars", "char_entropy", "n_tokens_target"]

_ARTIFACT_COLS = [
    "tok_rank_mean", "tok_rank_min", "tok_rank_max",
    "merge_rank_mean", "merge_rank_max",
    "piece_score_mean", "piece_score_min", "piece_score_max",
]

# Columns that must never appear as features: identity, labels, metadata strings,
# and ALL scoring-derived columns from build_labels / score_bpc.
# SPEC §6: features must use only tokenizer artifacts and the string s.
# total_bits_ref / total_bits_target are LM logprob sums — including them
# would give the predictor direct access to model output (label leakage).
_NON_FEATURE_COLS = frozenset([
    "candidate_id", "text_raw",
    "target_tok_id", "target_family", "target_normalizer_id",
    "target_norm", "extracted_at",
    # Label columns
    "delta_bpc", "bpc_ref", "bpc_target",
    "valid_ref", "valid_target", "valid_label",
    "invalid_reason_ref", "invalid_reason_target", "invalid_reason_label",
    "label_top_5pct", "label_top_1pct", "label_top_0_1pct",
    # Labels-parquet provenance
    "tok_id", "run_id",
    # Scoring-derived columns (from labels parquet passthrough of score_bpc).
    # total_bits are LM-derived — CRITICAL label leakage if included.
    # The rest are scoring metadata, not tokenizer-only features.
    "total_bits_ref", "total_bits_target",
    "n_tokens_ref", "len_chars_ref", "len_chars_target",
    "truncated_ref", "truncated_target",
    "scored_at_ref", "scored_at_target",
    "model_dir_ref", "model_dir_target",
    "normalized_text_ref", "normalized_text_target",
])

# Classification label columns, ordered broadest→narrowest.
# Only columns actually present in the parquet and non-all-NaN after valid
# filtering will be used.
_CLS_LABEL_COLS = ["label_top_5pct", "label_top_1pct", "label_top_0_1pct"]

# Regression target
_REG_TARGET_COL = "delta_bpc"

# Feature subsets: static lists or sentinels for dynamic resolution
# Ordered from trivial -> rich for ablation analysis
_STATIC_SUBSETS: Dict[str, Optional[List[str]]] = {
    "token_count_only":      ["n_tokens_target"],
    "len_entropy":           ["len_chars", "char_entropy"],
    "baseline_zlib":         ["len_chars", "char_entropy", "zlib_bpc", "zlib_compression_ratio"],
    "token_count_delta_tok": None,   # resolved dynamically
    "full":                  None,   # resolved dynamically
}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_parquet(path: str, label: str) -> Any:
    """Load a parquet file, fail loudly if not found."""
    import pandas as pd  # type: ignore[import-untyped]
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} file not found: {path!r}")
    df = pd.read_parquet(path)
    logger.info("Loaded %s: %d rows, %d cols from %s", label, len(df), len(df.columns), path)
    return df


def _join_features_labels(df_feat: Any, df_lab: Any) -> Any:
    """
    Join features and labels on candidate_id.

    Validates:
    - Both frames have candidate_id.
    - No duplicate candidate_ids in either frame.
    - ID sets are exactly equal (no missing/extra rows).
    Returns the merged DataFrame.
    """
    import pandas as pd  # type: ignore[import-untyped]

    for df, name in [(df_feat, "features"), (df_lab, "labels")]:
        if "candidate_id" not in df.columns:
            raise ValueError(f"{name} parquet is missing column 'candidate_id'")
        dupes = df["candidate_id"].value_counts()
        dupes = dupes[dupes > 1].index.tolist()
        if dupes:
            raise ValueError(
                f"{name} parquet has duplicate candidate_ids: {dupes[:5]}"
            )

    ids_feat = set(df_feat["candidate_id"])
    ids_lab = set(df_lab["candidate_id"])
    only_feat = sorted(ids_feat - ids_lab)
    only_lab = sorted(ids_lab - ids_feat)
    if only_feat or only_lab:
        msgs = []
        if only_feat:
            msgs.append(f"  {len(only_feat)} id(s) in features but not labels: {only_feat[:3]}")
        if only_lab:
            msgs.append(f"  {len(only_lab)} id(s) in labels but not features: {only_lab[:3]}")
        raise ValueError(
            "features and labels candidate_id sets do not match:\n" + "\n".join(msgs)
        )

    # Keep features-side order
    label_cols_to_merge = [c for c in df_lab.columns if c != "candidate_id"]
    # Drop columns already in features to avoid duplication (text_raw, tok_id, run_id)
    label_cols_to_merge = [
        c for c in label_cols_to_merge if c not in df_feat.columns
    ]
    df = df_feat.merge(
        df_lab[["candidate_id"] + label_cols_to_merge],
        on="candidate_id",
        how="inner",
    )

    if len(df) != len(df_feat):
        raise RuntimeError(
            f"Merge produced {len(df)} rows but expected {len(df_feat)}. Bug."
        )

    logger.info("Joined: %d rows total", len(df))
    return df


# ---------------------------------------------------------------------------
# Feature subset resolution
# ---------------------------------------------------------------------------

def _get_feature_cols(
    subset_name: str,
    all_cols: List[str],
    feature_origin_cols: Optional[frozenset] = None,
) -> List[str]:
    """
    Return the ordered list of feature column names for a given subset.

    Parameters
    ----------
    subset_name         : one of the four defined subsets
    all_cols            : all columns in the merged DataFrame
    feature_origin_cols : columns that originated from the features parquet
                          (required for 'full' subset to prevent label leakage)

    Columns that are not present in all_cols are silently skipped;
    the caller must check that the result is non-empty.
    """
    if subset_name == "token_count_only":
        candidates = ["n_tokens_target"]

    elif subset_name == "len_entropy":
        candidates = ["len_chars", "char_entropy"]

    elif subset_name == "baseline_zlib":
        candidates = ["len_chars", "char_entropy", "zlib_bpc", "zlib_compression_ratio"]

    elif subset_name == "token_count_delta_tok":
        ref_delta_cols = sorted(
            c for c in all_cols
            if c.startswith("ref_") and c.endswith("_delta_tok")
        )
        candidates = ["n_tokens_target"] + ref_delta_cols

    elif subset_name == "full":
        if feature_origin_cols is None:
            raise ValueError(
                "feature_origin_cols is required for 'full' subset "
                "to prevent label leakage from labels-parquet columns."
            )
        # Only columns that (a) originate from the features parquet,
        # (b) are not in the explicit exclusion set, and
        # (c) are not string metadata.
        # This prevents any labels/scoring column from ever being used,
        # even if _NON_FEATURE_COLS misses a future addition.
        candidates = [
            c for c in all_cols
            if c in feature_origin_cols
            and c not in _NON_FEATURE_COLS
            and not c.endswith("_normalizer_id")
        ]

    else:
        raise ValueError(f"Unknown feature subset: {subset_name!r}")

    present = [c for c in candidates if c in all_cols]
    missing = [c for c in candidates if c not in all_cols]
    if missing:
        logger.debug(
            "Subset %r: %d requested columns not present in data: %s",
            subset_name, len(missing), missing,
        )
    return present


# ---------------------------------------------------------------------------
# Train / val / test split
# ---------------------------------------------------------------------------

def _seeded_split(
    n: int, train_frac: float, val_frac: float, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (train_idx, val_idx, test_idx) for n items.

    Uses a seeded numpy permutation — no sklearn dependency.
    Floor-rounds train and val; test gets the remainder.
    """
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_train = int(math.floor(train_frac * n))
    n_val = int(math.floor(val_frac * n))
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]
    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Feature matrix preparation (imputation — NO leakage from val/test)
# ---------------------------------------------------------------------------

def _prepare_xy(
    df_train: Any,
    df_val: Any,
    df_test: Any,
    feature_cols: List[str],
    target_col: str,
) -> Tuple[
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    List[str], List[str], Any,
]:
    """
    Build X/y matrices for train, val, test.

    Steps:
    1. Cast to float.
    2. Drop columns that are entirely NaN in training split.
    3. Fit median imputer on training split only.
    4. Apply imputer to all three splits.

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test,
    used_cols, dropped_cols, imputer
    """
    from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]

    X_tr_raw = df_train[feature_cols].values.astype(float)
    X_va_raw = df_val[feature_cols].values.astype(float)
    X_te_raw = df_test[feature_cols].values.astype(float)

    # Drop columns entirely NaN in training — imputer can't handle them
    all_nan_mask = np.all(np.isnan(X_tr_raw), axis=0)
    if np.all(all_nan_mask):
        raise ValueError(
            f"All {len(feature_cols)} feature columns are entirely NaN in training "
            f"split for subset: {feature_cols}. Cannot train."
        )

    keep_mask = ~all_nan_mask
    used_cols = [c for c, k in zip(feature_cols, keep_mask) if k]
    dropped_cols = [c for c, k in zip(feature_cols, keep_mask) if not k]
    if dropped_cols:
        logger.warning(
            "Dropped %d entirely-NaN feature(s) from training split: %s",
            len(dropped_cols), dropped_cols,
        )

    X_tr = X_tr_raw[:, keep_mask]
    X_va = X_va_raw[:, keep_mask]
    X_te = X_te_raw[:, keep_mask]

    # Impute remaining NaN with training median (fit on train ONLY)
    imputer = SimpleImputer(strategy="median")
    X_tr = imputer.fit_transform(X_tr)
    X_va = imputer.transform(X_va)
    X_te = imputer.transform(X_te)

    # Verify no NaN in targets (defensive assertion — should be guaranteed upstream)
    y_tr = df_train[target_col].values.astype(float)
    y_va = df_val[target_col].values.astype(float)
    y_te = df_test[target_col].values.astype(float)

    for y, split in [(y_tr, "train"), (y_va, "val"), (y_te, "test")]:
        n_nan = int(np.isnan(y).sum())
        if n_nan > 0:
            raise ValueError(
                f"NaN found in target column {target_col!r} in {split} split "
                f"({n_nan} rows). This violates the no-NaN-label invariant. "
                f"Fix upstream (build_labels.py) before training."
            )

    return X_tr, y_tr, X_va, y_va, X_te, y_te, used_cols, dropped_cols, imputer


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def _build_clf(
    model_type: str,
    seed: int,
    max_iter: int,
    n_estimators: int,
    scale: bool = False,
) -> Any:
    """
    Build a classifier Pipeline.

    scale=True adds StandardScaler (needed for logistic regression).
    Imputation is intentionally NOT included here — done in _prepare_xy.
    """
    from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
    from sklearn.pipeline import Pipeline               # type: ignore[import-untyped]
    from sklearn.preprocessing import StandardScaler    # type: ignore[import-untyped]

    steps: list = []
    if scale:
        steps.append(("scale", StandardScaler()))

    if model_type == "logistic":
        steps.append((
            "clf",
            LogisticRegression(
                max_iter=max_iter,
                random_state=seed,
                C=1.0,
                solver="lbfgs",
                class_weight="balanced",
            ),
        ))
    elif model_type == "xgboost":
        try:
            from xgboost import XGBClassifier  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "xgboost is required for model_type='xgboost'. "
                "pip install xgboost"
            ) from e
        steps.append((
            "clf",
            XGBClassifier(
                n_estimators=n_estimators,
                random_state=seed,
                eval_metric="logloss",
                verbosity=0,
            ),
        ))
    elif model_type == "rf":
        from sklearn.ensemble import RandomForestClassifier  # type: ignore[import-untyped]
        steps.append((
            "clf",
            RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=seed,
                class_weight="balanced",
            ),
        ))
    else:
        raise ValueError(f"Unknown model_type for classification: {model_type!r}")

    return Pipeline(steps)


def _build_reg(
    model_type: str,
    seed: int,
    n_estimators: int,
    scale: bool = False,
) -> Any:
    """
    Build a regressor Pipeline.

    scale=True adds StandardScaler (needed for Ridge).
    Imputation is intentionally NOT included here — done in _prepare_xy.
    """
    from sklearn.pipeline import Pipeline             # type: ignore[import-untyped]
    from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

    steps: list = []
    if scale:
        steps.append(("scale", StandardScaler()))

    if model_type == "ridge":
        from sklearn.linear_model import Ridge  # type: ignore[import-untyped]
        steps.append(("reg", Ridge(alpha=1.0)))
    elif model_type == "xgboost":
        try:
            from xgboost import XGBRegressor  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "xgboost is required for model_type='xgboost'. "
                "pip install xgboost"
            ) from e
        steps.append((
            "reg",
            XGBRegressor(n_estimators=n_estimators, random_state=seed, verbosity=0),
        ))
    elif model_type == "rf":
        from sklearn.ensemble import RandomForestRegressor  # type: ignore[import-untyped]
        steps.append((
            "reg",
            RandomForestRegressor(n_estimators=n_estimators, random_state=seed),
        ))
    else:
        raise ValueError(f"Unknown model_type for regression: {model_type!r}")

    return Pipeline(steps)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """MSE, RMSE, Pearson r, Spearman rho."""
    from scipy.stats import pearsonr, spearmanr  # type: ignore[import-untyped]

    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))

    if len(y_true) < 2:
        return {"mse": mse, "rmse": rmse, "pearson_r": float("nan"), "spearman_rho": float("nan")}

    pr, _ = pearsonr(y_true, y_pred)
    sr, _ = spearmanr(y_true, y_pred)
    return {
        "mse": mse,
        "rmse": rmse,
        "pearson_r": float(pr),
        "spearman_rho": float(sr),
    }


def _classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    ranking_k: List[int],
) -> Dict[str, Any]:
    """
    AUROC, precision@K, TPR@FPR.

    Threshold is applied to convert scores to binary predictions.
    It must be chosen on val, NOT test (caller's responsibility).
    """
    from sklearn.metrics import (  # type: ignore[import-untyped]
        roc_auc_score,
        average_precision_score,
    )

    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    metrics: Dict[str, Any] = {
        "n_total": len(y_true),
        "n_positive": n_pos,
        "n_negative": n_neg,
    }

    if n_pos == 0 or n_neg == 0:
        logger.warning(
            "classification_metrics: only one class present "
            "(n_pos=%d, n_neg=%d). AUROC undefined.", n_pos, n_neg
        )
        metrics["auroc"] = float("nan")
        metrics["auprc"] = float("nan")
    else:
        metrics["auroc"] = float(roc_auc_score(y_true, y_score))
        metrics["auprc"] = float(average_precision_score(y_true, y_score))

    # Precision@K using sorted scores
    sort_idx = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[sort_idx]

    prec_at_k: Dict[str, float] = {}
    for k in ranking_k:
        if k > len(y_true):
            prec_at_k[f"precision_at_{k}"] = float("nan")
        else:
            prec_at_k[f"precision_at_{k}"] = float(
                y_true_sorted[:k].sum() / k
            )
    metrics["precision_at_k"] = prec_at_k

    # Binary metrics at configured threshold (threshold must be set on val, not test)
    y_pred_bin = (y_score >= threshold).astype(int)
    tp = int(((y_pred_bin == 1) & (y_true == 1)).sum())
    fp = int(((y_pred_bin == 1) & (y_true == 0)).sum())
    fn = int(((y_pred_bin == 0) & (y_true == 1)).sum())
    metrics["threshold_used"] = threshold
    metrics["tp"] = tp
    metrics["fp"] = fp
    metrics["fn"] = fn
    metrics["precision"] = float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
    metrics["recall"] = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")

    return metrics


def _choose_threshold_on_val(
    y_val: np.ndarray, y_score_val: np.ndarray
) -> float:
    """
    Select classification threshold on validation set.

    Picks the threshold that maximises F1 on the validation set.
    Must only be called with validation data — NEVER with test data.
    Returns the chosen threshold (float in [0, 1]).
    """
    from sklearn.metrics import f1_score  # type: ignore[import-untyped]

    if len(y_val) < 2 or int(y_val.sum()) == 0:
        logger.warning("Val set has no positive examples; keeping default threshold 0.5")
        return 0.5

    best_f1 = -1.0
    best_thr = 0.5
    for thr in np.linspace(0.1, 0.9, 17):
        y_pred = (y_score_val >= thr).astype(int)
        try:
            f1 = f1_score(y_val, y_pred, zero_division=0)
        except Exception:
            continue
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)

    logger.info("Threshold selected on val: %.3f  (best F1=%.4f)", best_thr, best_f1)
    return best_thr


# ---------------------------------------------------------------------------
# Single-configuration trainer
# ---------------------------------------------------------------------------

def _train_one_config(
    task: str,
    target_col: str,
    model_name: str,
    subset_name: str,
    df_train: Any,
    df_val: Any,
    df_test: Any,
    feature_cols_raw: List[str],
    output_dir: str,
    seed: int,
    max_iter: int,
    n_estimators: int,
    default_threshold: float,
    ranking_k: List[int],
) -> Dict[str, Any]:
    """
    Train and evaluate one (task, model, subset) combination.

    Saves model.pkl, predictions.parquet, val_metrics.json,
    test_metrics.json, and metadata.json to output_dir.

    Returns a metadata dict summarising the run.
    """
    import pandas as pd          # type: ignore[import-untyped]
    import joblib                # type: ignore[import-untyped]

    run_dir = output_dir
    os.makedirs(run_dir, exist_ok=True)

    logger.info(
        "  [%s | %s | %s]  n_train=%d  n_val=%d  n_test=%d",
        task, model_name, subset_name,
        len(df_train), len(df_val), len(df_test),
    )

    # --- Prepare feature matrices ---
    feature_cols = [c for c in feature_cols_raw if c in df_train.columns]
    if not feature_cols:
        raise ValueError(
            f"No feature columns available for subset={subset_name!r} "
            f"in the joined DataFrame."
        )

    X_tr, y_tr, X_va, y_va, X_te, y_te, used_cols, dropped_cols, imputer = \
        _prepare_xy(df_train, df_val, df_test, feature_cols, target_col)

    # --- Build and fit model ---
    is_clf = task == "classification"

    if is_clf:
        needs_scale = model_name == "logistic"
        model = _build_clf(model_name, seed, max_iter, n_estimators, scale=needs_scale)
        model.fit(X_tr, y_tr.astype(int))

        y_score_tr = model.predict_proba(X_tr)[:, 1]
        y_score_va = model.predict_proba(X_va)[:, 1]
        y_score_te = model.predict_proba(X_te)[:, 1]

        # Threshold selection on val ONLY
        chosen_threshold = _choose_threshold_on_val(y_va.astype(int), y_score_va)

        val_metrics = _classification_metrics(
            y_va.astype(int), y_score_va, chosen_threshold, ranking_k
        )
        test_metrics = _classification_metrics(
            y_te.astype(int), y_score_te, chosen_threshold, ranking_k
        )

        # Predictions parquet
        all_dfs = []
        for df_split, ids_col, y_t, y_s, split_name in [
            (df_train, df_train["candidate_id"], y_tr, y_score_tr, "train"),
            (df_val,   df_val["candidate_id"],   y_va, y_score_va, "val"),
            (df_test,  df_test["candidate_id"],  y_te, y_score_te, "test"),
        ]:
            all_dfs.append(pd.DataFrame({
                "candidate_id": ids_col.values,
                "split":        split_name,
                "y_true":       y_t,
                "y_score":      y_s,
                "y_pred_class": (y_s >= chosen_threshold).astype(int),
            }))

    else:  # regression
        needs_scale = model_name in ("ridge",)
        model = _build_reg(model_name, seed, n_estimators, scale=needs_scale)
        model.fit(X_tr, y_tr)

        y_pred_tr = model.predict(X_tr)
        y_pred_va = model.predict(X_va)
        y_pred_te = model.predict(X_te)

        val_metrics = _regression_metrics(y_va, y_pred_va)
        test_metrics = _regression_metrics(y_te, y_pred_te)

        chosen_threshold = default_threshold  # not used for regression

        all_dfs = []
        for ids_col, y_t, y_p, split_name in [
            (df_train["candidate_id"], y_tr, y_pred_tr, "train"),
            (df_val["candidate_id"],   y_va, y_pred_va, "val"),
            (df_test["candidate_id"],  y_te, y_pred_te, "test"),
        ]:
            all_dfs.append(pd.DataFrame({
                "candidate_id": ids_col.values,
                "split":        split_name,
                "y_true":       y_t,
                "y_pred":       y_p,
            }))

    # --- Save model ---
    model_path = os.path.join(run_dir, "model.pkl")
    joblib.dump({"model": model, "imputer": imputer, "used_cols": used_cols}, model_path)

    # --- Save predictions ---
    pred_df = pd.concat(all_dfs, ignore_index=True)
    pred_path = os.path.join(run_dir, "predictions.parquet")
    pred_df.to_parquet(pred_path, index=False, engine="pyarrow")

    # --- Save metrics ---
    val_metrics_path = os.path.join(run_dir, "val_metrics.json")
    test_metrics_path = os.path.join(run_dir, "test_metrics.json")
    with open(val_metrics_path, "w", encoding="utf-8") as fh:
        json.dump(val_metrics, fh, indent=2)
    with open(test_metrics_path, "w", encoding="utf-8") as fh:
        json.dump(test_metrics, fh, indent=2)

    # --- Save metadata ---
    metadata = {
        "task": task,
        "target_col": target_col,
        "model_name": model_name,
        "feature_subset": subset_name,
        "n_train": len(df_train),
        "n_val": len(df_val),
        "n_test": len(df_test),
        "n_features_requested": len(feature_cols),
        "n_features_used": len(used_cols),
        "features_used": used_cols,
        "features_dropped_all_nan": dropped_cols,
        "threshold_used": chosen_threshold,
        "threshold_selected_on": "val" if is_clf else "n/a",
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "trained_at": datetime.datetime.utcnow().isoformat() + "Z",
    }
    meta_path = os.path.join(run_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    logger.info(
        "    Saved -> %s  [val_metrics: %s]",
        run_dir,
        ", ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()
                  if isinstance(v, float) and not math.isnan(v)),
    )
    return metadata


# ---------------------------------------------------------------------------
# Main training orchestrator
# ---------------------------------------------------------------------------

def train_predictor(
    features_path: str,
    labels_path: str,
    output_dir: str,
    run_id: str,
    *,
    model_type: str = "logistic",
    split_train: float = 0.6,
    split_val: float = 0.2,
    seed: int = 42,
    max_iter: int = 500,
    n_estimators: int = 100,
    threshold: float = 0.5,
    ranking_k: Optional[List[int]] = None,
) -> str:
    """
    Train memorization risk predictors for all feature subsets × tasks.

    Parameters
    ----------
    features_path  : path to extract_features output parquet
    labels_path    : path to build_labels output parquet
    output_dir     : root directory for all outputs
    run_id         : experiment identifier
    model_type     : "logistic" | "xgboost" | "rf"
    split_train    : fraction of valid rows for training
    split_val      : fraction of valid rows for validation
                     (test gets the remainder: 1 - train - val)
    seed           : random seed for split + model
    max_iter       : LogisticRegression max_iter
    n_estimators   : tree ensemble n_estimators
    threshold      : default classification threshold (overridden by val selection)
    ranking_k      : list of K values for precision@K [default: [50, 100, 500]]

    Returns
    -------
    str  Path to the top-level manifest JSON.
    """
    import pandas as pd  # type: ignore[import-untyped]

    if ranking_k is None:
        ranking_k = [50, 100, 500]

    if split_train + split_val >= 1.0:
        raise ValueError(
            f"split_train ({split_train}) + split_val ({split_val}) >= 1.0. "
            f"No room for a test split."
        )

    logger.info("=" * 65)
    logger.info("train_predictor: run_id=%s  model=%s", run_id, model_type)
    logger.info("=" * 65)

    # ------------------------------------------------------------------
    # Load + join
    # ------------------------------------------------------------------
    df_feat = _load_parquet(features_path, "features")
    df_lab = _load_parquet(labels_path, "labels")
    df = _join_features_labels(df_feat, df_lab)

    # ------------------------------------------------------------------
    # Filter to valid_label=True — this is the only pool for all tasks
    # ------------------------------------------------------------------
    if "valid_label" not in df.columns:
        raise ValueError(
            "Joined DataFrame is missing 'valid_label' column. "
            "Ensure labels parquet is from build_labels.py."
        )

    n_total = len(df)
    df_valid = df[df["valid_label"].astype(bool)].reset_index(drop=True)
    n_valid = len(df_valid)
    n_excluded = n_total - n_valid

    logger.info(
        "Valid rows: %d / %d  (%d excluded as invalid)", n_valid, n_total, n_excluded
    )

    if n_valid < 3:
        raise ValueError(
            f"Only {n_valid} valid rows after filtering. Need at least 3 "
            f"(one per split) to train a predictor."
        )

    # ------------------------------------------------------------------
    # Assert no NaN in delta_bpc for valid rows (upstream invariant)
    # ------------------------------------------------------------------
    if _REG_TARGET_COL not in df_valid.columns:
        raise ValueError(
            f"Column {_REG_TARGET_COL!r} not found in joined DataFrame."
        )
    n_nan_delta = int(df_valid[_REG_TARGET_COL].isna().sum())
    if n_nan_delta > 0:
        raise ValueError(
            f"{n_nan_delta} valid rows have NaN delta_bpc. "
            "This violates the upstream invariant from build_labels.py."
        )

    # ------------------------------------------------------------------
    # Determine which classification tasks are runnable
    # ------------------------------------------------------------------
    cls_tasks = []
    for col in _CLS_LABEL_COLS:
        if col not in df_valid.columns:
            logger.info("Classification label %r not in data — skipping.", col)
            continue
        n_nan = int(df_valid[col].isna().sum())
        if n_nan > 0:
            logger.warning(
                "Classification label %r has %d NaN values in valid rows "
                "(not resolvable per build_labels.py). Skipping this task.",
                col, n_nan,
            )
            continue
        n_pos = int((df_valid[col] == 1.0).sum())
        if n_pos == 0:
            logger.warning(
                "Classification label %r has 0 positive examples in valid rows. "
                "Skipping (cannot train meaningful classifier).", col
            )
            continue
        cls_tasks.append(col)
        logger.info(
            "Classification task: %s  (n_pos=%d / %d)", col, n_pos, n_valid
        )

    # ------------------------------------------------------------------
    # Single seeded split — shared across all tasks and subsets
    # ------------------------------------------------------------------
    train_idx, val_idx, test_idx = _seeded_split(n_valid, split_train, split_val, seed)

    df_train = df_valid.iloc[train_idx].reset_index(drop=True)
    df_val   = df_valid.iloc[val_idx].reset_index(drop=True)
    df_test  = df_valid.iloc[test_idx].reset_index(drop=True)

    logger.info(
        "Split: %d train / %d val / %d test  (seed=%d)",
        len(df_train), len(df_val), len(df_test), seed,
    )

    if len(df_val) == 0 or len(df_test) == 0:
        raise ValueError(
            f"Val ({len(df_val)}) or test ({len(df_test)}) split is empty. "
            f"Increase n_valid ({n_valid}) or adjust split fractions."
        )

    # Save split manifest
    os.makedirs(output_dir, exist_ok=True)
    split_manifest = {
        "run_id": run_id,
        "seed": seed,
        "split_fracs": {"train": split_train, "val": split_val, "test": 1.0 - split_train - split_val},
        "n_valid_total": n_valid,
        "n_train": len(df_train),
        "n_val": len(df_val),
        "n_test": len(df_test),
        "train_ids": df_train["candidate_id"].tolist(),
        "val_ids":   df_val["candidate_id"].tolist(),
        "test_ids":  df_test["candidate_id"].tolist(),
    }
    split_path = os.path.join(output_dir, f"{run_id}_split.json")
    with open(split_path, "w", encoding="utf-8") as fh:
        json.dump(split_manifest, fh, indent=2)
    logger.info("Split manifest -> %s", split_path)

    # ------------------------------------------------------------------
    # Build list of all (task, target_col, model_name) combinations
    # ------------------------------------------------------------------
    all_cols = list(df_valid.columns)
    feature_origin_cols = frozenset(df_feat.columns)
    run_configs: List[Tuple[str, str, str]] = []

    # Regression: always run Ridge baseline + configured model (if different)
    run_configs.append(("regression", _REG_TARGET_COL, "ridge"))
    if model_type not in ("logistic",) and model_type != "ridge":
        # logistic doesn't make sense for regression
        run_configs.append(("regression", _REG_TARGET_COL, model_type))

    # Classification: configured model for each resolvable label
    for cls_col in cls_tasks:
        run_configs.append(("classification", cls_col, model_type))

    # ------------------------------------------------------------------
    # Run all (task, target_col, model_name) × feature_subset combinations
    # ------------------------------------------------------------------
    all_run_metadata: List[Dict[str, Any]] = []
    all_subset_names = list(_STATIC_SUBSETS.keys())

    for task, target_col, model_name in run_configs:
        for subset_name in all_subset_names:
            feature_cols = _get_feature_cols(
                subset_name, all_cols, feature_origin_cols=feature_origin_cols,
            )
            if not feature_cols:
                logger.warning(
                    "Skipping %s/%s/%s: no feature columns available.",
                    task, model_name, subset_name,
                )
                continue

            # Directory name: task_targetshort_model_subset
            target_short = target_col.replace("label_", "").replace("delta_bpc", "delta_bpc")
            dir_name = f"{task}_{target_short}_{model_name}_{subset_name}"
            run_dir = os.path.join(output_dir, dir_name)

            try:
                meta = _train_one_config(
                    task=task,
                    target_col=target_col,
                    model_name=model_name,
                    subset_name=subset_name,
                    df_train=df_train,
                    df_val=df_val,
                    df_test=df_test,
                    feature_cols_raw=feature_cols,
                    output_dir=run_dir,
                    seed=seed,
                    max_iter=max_iter,
                    n_estimators=n_estimators,
                    default_threshold=threshold,
                    ranking_k=ranking_k,
                )
                meta["run_dir"] = run_dir
                all_run_metadata.append(meta)

            except Exception as e:
                logger.error(
                    "FAILED: %s/%s/%s — %s", task, model_name, subset_name, e
                )
                all_run_metadata.append({
                    "task": task,
                    "target_col": target_col,
                    "model_name": model_name,
                    "feature_subset": subset_name,
                    "run_dir": run_dir,
                    "error": str(e),
                })

    # ------------------------------------------------------------------
    # Write top-level manifest
    # ------------------------------------------------------------------
    manifest = {
        "run_id": run_id,
        "built_at": datetime.datetime.utcnow().isoformat() + "Z",
        "inputs": {
            "features": os.path.abspath(features_path),
            "labels": os.path.abspath(labels_path),
        },
        "config": {
            "model_type": model_type,
            "split_train": split_train,
            "split_val": split_val,
            "seed": seed,
            "max_iter": max_iter,
            "n_estimators": n_estimators,
            "threshold": threshold,
            "ranking_k": ranking_k,
        },
        "row_counts": {
            "total": n_total,
            "valid": n_valid,
            "excluded_invalid": n_excluded,
            "train": len(df_train),
            "val": len(df_val),
            "test": len(df_test),
        },
        "classification_tasks": cls_tasks,
        "regression_task": _REG_TARGET_COL,
        "runs": all_run_metadata,
    }

    manifest_path = os.path.join(output_dir, f"{run_id}_predictor_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    n_ok = sum(1 for r in all_run_metadata if "error" not in r)
    n_fail = len(all_run_metadata) - n_ok
    logger.info(
        "Completed %d / %d configs (%d failed) -> manifest: %s",
        n_ok, len(all_run_metadata), n_fail, manifest_path,
    )
    logger.info("=" * 65)
    return manifest_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.train_predictor",
        description="Train memorization risk predictors (all subsets × tasks).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", metavar="YAML", help="Project YAML config.")
    p.add_argument("--features", metavar="FILE", help="Features parquet.")
    p.add_argument("--labels", metavar="FILE", help="Labels parquet.")
    p.add_argument("--output", metavar="DIR", help="Output directory.")
    p.add_argument("--run-id", metavar="ID")
    p.add_argument("--model-type", choices=["logistic", "xgboost", "rf"])
    p.add_argument("--seed", type=int)
    p.add_argument("--max-iter", type=int)
    p.add_argument("--n-estimators", type=int)
    p.add_argument("--threshold", type=float)
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

    features = labels = output_dir = run_id = None
    model_type = "logistic"
    seed = 42
    split_train, split_val = 0.6, 0.2
    max_iter = 500
    n_estimators = 100
    threshold = 0.5
    ranking_k: Optional[List[int]] = None

    if args.config:
        from src.config import load_config  # type: ignore[import]
        cfg = load_config(args.config)
        run_id = cfg.run_id
        seed = cfg.seed
        features = os.path.join(cfg.paths.features_dir, f"{run_id}.parquet")
        labels = os.path.join(cfg.paths.labels_dir, f"{run_id}_labels.parquet")
        output_dir = os.path.join(cfg.paths.results_dir, run_id, "predictor")
        model_type = cfg.predictor.model_type
        split_train = cfg.predictor.split.train
        split_val = cfg.predictor.split.val
        max_iter = cfg.predictor.max_iter
        n_estimators = cfg.predictor.n_estimators
        threshold = cfg.predictor.threshold
        ranking_k = cfg.evaluation.ranking_k

    if args.features:    features = args.features
    if args.labels:      labels = args.labels
    if args.output:      output_dir = args.output
    if args.run_id:      run_id = args.run_id
    if args.model_type:  model_type = args.model_type
    if args.seed:        seed = args.seed
    if args.max_iter:    max_iter = args.max_iter
    if args.n_estimators: n_estimators = args.n_estimators
    if args.threshold:   threshold = args.threshold

    missing = [
        name for name, val in [
            ("--features", features),
            ("--labels",   labels),
            ("--output",   output_dir),
            ("--run-id",   run_id),
        ]
        if val is None
    ]
    if missing:
        parser.error("Missing required arguments:\n  " + "\n  ".join(missing))

    try:
        out = train_predictor(
            features_path=features,          # type: ignore[arg-type]
            labels_path=labels,              # type: ignore[arg-type]
            output_dir=output_dir,           # type: ignore[arg-type]
            run_id=run_id,                   # type: ignore[arg-type]
            model_type=model_type,
            split_train=split_train,
            split_val=split_val,
            seed=seed,
            max_iter=max_iter,
            n_estimators=n_estimators,
            threshold=threshold,
            ranking_k=ranking_k,
        )
    except (FileNotFoundError, ValueError, RuntimeError, ImportError) as e:
        logger.error("FATAL: %s", e)
        sys.exit(1)

    print(f"\nDone.  Manifest written to: {out}")


if __name__ == "__main__":
    main()
