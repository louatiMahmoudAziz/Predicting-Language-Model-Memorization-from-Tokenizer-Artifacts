"""
Predicting Language Model Memorization from Tokenizer Artifacts
================================================================
Redesigned experiment using EleutherAI's published memorization metric
from "Emergent and Predictable Memorization in Large Language Models"
(Biderman et al., NeurIPS 2023).

Labels: per-span greedy continuation accuracy — fraction of 32
continuation tokens exactly reproduced via greedy decoding from a
32-token prefix.  This is an extraction-based memorization metric
under a specific protocol, not a universal ground truth.

Dataset: EleutherAI/pile-deduped-pythia-random-sampled (5M spans).

Features: tokenizer-only features in ablation groups
(trivial → baseline → counts → full → full+freq).
"""

import os
import math
import json
import zlib
import copy
import logging
import argparse
import warnings
from typing import Dict, List, Optional
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PREFIX_LEN = 32  # memorization score uses first 32 tokens as prefix

# ============================================================
# 1. Dataset loading (streaming-safe)
# ============================================================

def load_memorization_dataset(
    n_samples: Optional[int] = None,
    seed: int = 42,
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Load EleutherAI/pile-deduped-pythia-random-sampled from HuggingFace.

    Streams and samples before converting to pandas to avoid
    loading all 5M list-of-int rows into memory at once.
    """
    from datasets import load_dataset

    logger.info("Loading EleutherAI/pile-deduped-pythia-random-sampled ...")

    if n_samples is not None:
        ds = load_dataset(
            "EleutherAI/pile-deduped-pythia-random-sampled",
            split=f"train[:{min(n_samples * 3, 5_000_000)}]",
            cache_dir=cache_dir,
        )
        df = ds.to_pandas()
        if len(df) > n_samples:
            df = df.sample(n=n_samples, random_state=seed).reset_index(drop=True)
    else:
        ds = load_dataset(
            "EleutherAI/pile-deduped-pythia-random-sampled",
            split="train",
            cache_dir=cache_dir,
        )
        df = ds.to_pandas()

    logger.info("Dataset: %d rows, columns: %s", len(df), list(df.columns))
    return df


# ============================================================
# 2. Tokenizer feature extraction
# ============================================================

def _char_entropy(s: str) -> float:
    if not s:
        return float("nan")
    freq = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def _zlib_bpc(s: str) -> float:
    if not s:
        return float("nan")
    raw = s.encode("utf-8")
    compressed = zlib.compress(raw, level=9)
    return (len(compressed) * 8) / len(s)


def _norm_id_rank_stats(token_ids: List[int], vocab_size: int) -> Tuple[float, float, float]:
    """Fallback when BPE merge ranks are unavailable: ID / (V-1) in [0, 1]."""
    denom = max(vocab_size - 1, 1)
    pcts = [tid / denom for tid in token_ids]
    return float(np.mean(pcts)), float(np.min(pcts)), float(np.max(pcts))


def impute_nan_median_train_test(
    X_train: np.ndarray, X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Impute NaNs with per-column training medians (0 if column all-NaN in train)."""
    med = np.nanmedian(X_train, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Xtr = np.where(np.isnan(X_train), med, X_train)
    Xte = np.where(np.isnan(X_test), med, X_test)
    Xtr = np.nan_to_num(Xtr, nan=0.0, posinf=0.0, neginf=0.0)
    Xte = np.nan_to_num(Xte, nan=0.0, posinf=0.0, neginf=0.0)
    return Xtr.astype(np.float64), Xte.astype(np.float64)


def impute_nan_median_columns(X: np.ndarray) -> np.ndarray:
    """Column-wise median imputation for a single matrix (e.g. SHAP, scale curve)."""
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    X2 = np.where(np.isnan(X), med, X)
    return np.nan_to_num(X2, nan=0.0, posinf=0.0, neginf=0.0)


def _build_merge_rank_map(tokenizer) -> Optional[Dict[int, int]]:
    """Build token_id → merge_rank from BPE merges file if available."""
    merges = getattr(tokenizer, "bpe_ranks", None)
    if merges is not None:
        token2rank = {}
        for pair, rank in merges.items():
            merged = "".join(pair)
            tid = tokenizer.convert_tokens_to_ids(merged)
            if tid is not None and tid != tokenizer.unk_token_id:
                token2rank[tid] = rank
        if token2rank:
            return token2rank

    if hasattr(tokenizer, "backend_tokenizer"):
        bt = tokenizer.backend_tokenizer
        model = getattr(bt, "model", None)
        if model is not None:
            merges_list = getattr(model, "merges", None) or []
            if not merges_list:
                try:
                    model_json = json.loads(bt.to_str())
                    merges_list = model_json.get("model", {}).get("merges", [])
                except Exception:
                    merges_list = []
            if merges_list:
                token2rank = {}
                for rank, merge_str in enumerate(merges_list):
                    parts = merge_str.split(" ") if isinstance(merge_str, str) else merge_str
                    merged = "".join(parts)
                    tid = tokenizer.convert_tokens_to_ids(merged)
                    if tid is not None and tid != getattr(tokenizer, "unk_token_id", -1):
                        token2rank[tid] = rank
                if token2rank:
                    return token2rank
    return None


def extract_features(
    token_lists: List[List[int]],
    tokenizer,
    batch_label: str = "matched",
    prefix_only: bool = True,
    merge_rank_map: Optional[Dict[int, int]] = None,
) -> pd.DataFrame:
    """Extract tokenizer-only features for each token sequence.

    If prefix_only=True, features are computed from the first
    PREFIX_LEN tokens only (matching the memorization evaluation
    protocol, which uses a 32-token prefix for generation).

    Features (cumulative groups):
      trivial(1):  len_chars
      baseline(3): + char_entropy, zlib_bpc
      length_only(4): + n_tokens  (separates length from structure)
      counts(6):   + compression_ratio, log_token_count
      full(12):    + merge_rank_{mean,min,max}, max_token_len,
                     mean_token_len, frac_single_char_tokens
    """
    vocab_size = tokenizer.vocab_size
    max_merge = max(merge_rank_map.values()) if merge_rank_map else 1
    special_ids = set()
    for attr in ["eos_token_id", "bos_token_id", "pad_token_id", "unk_token_id"]:
        tid = getattr(tokenizer, attr, None)
        if tid is not None:
            special_ids.add(tid)

    rows = []
    n_invalid_ids = 0

    for i, full_token_ids in enumerate(token_lists):
        token_ids = full_token_ids[:PREFIX_LEN] if prefix_only else full_token_ids
        row: Dict = {}

        n_special = sum(1 for t in token_ids if t in special_ids)
        non_special_ids = [t for t in token_ids if t not in special_ids]

        text = tokenizer.decode(non_special_ids, skip_special_tokens=True)

        bad_ids = sum(1 for t in token_ids if t >= vocab_size)
        n_invalid_ids += bad_ids

        n_chars = len(text)
        n_tokens = len(token_ids)
        n_content_tokens = len(non_special_ids)

        row["len_chars"] = n_chars
        row["char_entropy"] = _char_entropy(text)
        row["zlib_bpc"] = _zlib_bpc(text)
        row["n_tokens"] = n_tokens
        row["n_special_tokens"] = n_special
        row["compression_ratio"] = n_content_tokens / n_chars if n_chars > 0 else float("nan")
        row["log_token_count"] = math.log(n_content_tokens) if n_content_tokens > 0 else float("nan")

        if n_content_tokens > 0:
            if merge_rank_map:
                ranks = [merge_rank_map.get(tid, max_merge) for tid in non_special_ids]
                normed = [r / max_merge for r in ranks]
                row["merge_rank_mean"] = float(np.mean(normed))
                row["merge_rank_min"] = float(np.min(normed))
                row["merge_rank_max"] = float(np.max(normed))
            else:
                rm, rmn, rmx = _norm_id_rank_stats(non_special_ids, vocab_size)
                row["merge_rank_mean"] = rm
                row["merge_rank_min"] = rmn
                row["merge_rank_max"] = rmx

            pieces = tokenizer.convert_ids_to_tokens(non_special_ids)
            piece_lens = [len(p) for p in pieces]
            row["max_token_len"] = max(piece_lens)
            row["mean_token_len"] = float(np.mean(piece_lens))
            row["frac_single_char_tokens"] = sum(1 for pl in piece_lens if pl == 1) / n_content_tokens
        else:
            for col in ["merge_rank_mean", "merge_rank_min", "merge_rank_max",
                         "max_token_len", "mean_token_len", "frac_single_char_tokens"]:
                row[col] = float("nan")

        rows.append(row)

        if (i + 1) % 50000 == 0:
            logger.info("[%s] Extracted features: %d / %d", batch_label, i + 1, len(token_lists))

    if n_invalid_ids > 0:
        logger.warning("[%s] %d token IDs >= vocab_size (%d) detected",
                       batch_label, n_invalid_ids, vocab_size)

    logger.info("[%s] Feature extraction complete: %d spans (prefix_only=%s)",
                batch_label, len(rows), prefix_only)
    return pd.DataFrame(rows)


# ============================================================
# 3. Feature groups for ablation
# ============================================================

FEATURE_GROUPS = {
    "trivial": ["len_chars"],
    "baseline": ["len_chars", "char_entropy", "zlib_bpc"],
    "length_only": ["len_chars", "char_entropy", "zlib_bpc", "n_tokens"],
    "counts": ["len_chars", "char_entropy", "zlib_bpc",
                "n_tokens", "compression_ratio", "log_token_count"],
    "full": ["len_chars", "char_entropy", "zlib_bpc",
             "n_tokens", "compression_ratio", "log_token_count",
             "merge_rank_mean", "merge_rank_min", "merge_rank_max",
             "max_token_len", "mean_token_len", "frac_single_char_tokens"],
}

ALL_FEATURE_COLS = FEATURE_GROUPS["full"]

MODEL_SIZES = ["70M", "160M", "410M", "1B", "1.4B", "2.8B", "6.9B", "12B"]


# ============================================================
# 4. Labeling
# ============================================================

def make_binary_label(scores: np.ndarray, threshold_pct: Optional[float] = None,
                      threshold_abs: Optional[float] = None) -> np.ndarray:
    """Create binary labels from continuous memorization scores.

    Either use a relative percentile threshold (top k%) or an
    absolute threshold (score >= value).  Handles ties by using
    'higher' interpolation so the positive class is at most k%.
    """
    if threshold_abs is not None:
        return (scores >= threshold_abs).astype(int)

    if threshold_pct is not None:
        threshold = np.percentile(scores, 100 - threshold_pct,
                                  interpolation="higher")
        labels = (scores >= threshold).astype(int)
        actual_pct = 100 * labels.mean()
        if abs(actual_pct - threshold_pct) > 1.0:
            logger.warning(
                "Requested top-%.1f%% but got %.1f%% due to ties (threshold=%.4f)",
                threshold_pct, actual_pct, threshold,
            )
        return labels

    raise ValueError("Must specify threshold_pct or threshold_abs")


def get_label_configs(threshold_pcts: List[float]) -> List[Dict]:
    """Build list of labeling configs: percentile-based + absolute thresholds."""
    configs = []
    for pct in threshold_pcts:
        configs.append({"name": f"top_{pct}pct", "threshold_pct": pct, "threshold_abs": None})
    configs.append({"name": "exact_match", "threshold_pct": None, "threshold_abs": 1.0})
    configs.append({"name": "half_match", "threshold_pct": None, "threshold_abs": 0.5})
    return configs


# ============================================================
# 5. Evaluation: CV + ablation
# ============================================================

def run_single_fold(
    X_train: np.ndarray, y_train_cls: np.ndarray, y_train_reg: np.ndarray,
    X_test: np.ndarray, y_test_cls: np.ndarray, y_test_reg: np.ndarray,
    group_name: str, feature_names: List[str],
    use_gbm: bool = False,
) -> Dict:
    """Train + evaluate on one fold for one feature group."""
    Xtr, Xte = impute_nan_median_train_test(X_train, X_test)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xte = scaler.transform(Xte)

    result = {
        "group": group_name,
        "n_features": len(feature_names),
        "n_train": len(Xtr),
        "n_test": len(Xte),
        "n_pos_test": int(y_test_cls.sum()),
    }

    if use_gbm:
        model_name = "gbm"
        clf = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42,
        )
    else:
        model_name = "logistic"
        clf = LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42, C=1.0,
        )
    result["model"] = model_name

    n_pos_train = int(y_train_cls.sum())
    if n_pos_train < 2 or (len(y_train_cls) - n_pos_train) < 2:
        result.update({"auroc": float("nan"), "auprc": float("nan"),
                       "spearman_rho": float("nan"), "spearman_pval": float("nan")})
        return result

    clf.fit(Xtr, y_train_cls)

    y_prob = clf.predict_proba(Xte)[:, 1] if hasattr(clf, "predict_proba") \
        else clf.decision_function(Xte)

    n_pos_test = int(y_test_cls.sum())
    n_neg_test = len(y_test_cls) - n_pos_test
    if n_pos_test > 0 and n_neg_test > 0:
        result["auroc"] = roc_auc_score(y_test_cls, y_prob)
        result["auprc"] = average_precision_score(y_test_cls, y_prob)
    else:
        result["auroc"] = float("nan")
        result["auprc"] = float("nan")

    ridge = Ridge(alpha=1.0)
    ridge.fit(Xtr, y_train_reg)
    y_pred_reg = ridge.predict(Xte)
    rho, pval = spearmanr(y_test_reg, y_pred_reg)
    result["spearman_rho"] = rho
    result["spearman_pval"] = pval

    return result


def run_cv_ablation(
    features_df: pd.DataFrame,
    memorization_scores: np.ndarray,
    target_model: str,
    label_config: Dict,
    n_folds: int = 5,
    n_repeats: int = 3,
    seed: int = 42,
    include_random_null: bool = True,
) -> pd.DataFrame:
    """Run cross-validated ablation across feature groups.

    Includes a random-noise null baseline to verify AUROC
    improvements are not due to overfitting.
    """
    X_all = features_df[ALL_FEATURE_COLS].values
    y_reg = memorization_scores.copy()
    y_cls = make_binary_label(y_reg, threshold_pct=label_config.get("threshold_pct"),
                              threshold_abs=label_config.get("threshold_abs"))

    logger.info(
        "CV ablation: target=%s, label=%s, n=%d, pos=%d (%.2f%%)",
        target_model, label_config["name"], len(y_cls), y_cls.sum(),
        100 * y_cls.mean(),
    )

    if y_cls.sum() < 10:
        logger.warning("Too few positives (%d) for label %s — skipping",
                       y_cls.sum(), label_config["name"])
        return pd.DataFrame()

    all_results = []

    groups_to_run = dict(FEATURE_GROUPS)
    if include_random_null:
        groups_to_run["random_null"] = ALL_FEATURE_COLS  # same dim, random values

    for repeat in range(n_repeats):
        repeat_seed = seed + repeat * 1000
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=repeat_seed)

        rng = np.random.RandomState(repeat_seed)
        X_random = rng.randn(*X_all.shape)

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_all, y_cls)):
            y_train_cls, y_test_cls = y_cls[train_idx], y_cls[test_idx]
            y_train_reg, y_test_reg = y_reg[train_idx], y_reg[test_idx]

            for group_name, feature_cols in groups_to_run.items():
                if group_name == "random_null":
                    X_train = X_random[train_idx]
                    X_test = X_random[test_idx]
                else:
                    col_indices = [ALL_FEATURE_COLS.index(c) for c in feature_cols]
                    X_train = X_all[train_idx][:, col_indices]
                    X_test = X_all[test_idx][:, col_indices]

                res = run_single_fold(
                    X_train, y_train_cls, y_train_reg,
                    X_test, y_test_cls, y_test_reg,
                    group_name, feature_cols,
                    use_gbm=False,
                )
                res["repeat"] = repeat
                res["fold"] = fold_idx
                res["target_model"] = target_model
                res["label"] = label_config["name"]
                all_results.append(res)

            # GBM on full features only
            col_indices = [ALL_FEATURE_COLS.index(c) for c in FEATURE_GROUPS["full"]]
            res = run_single_fold(
                X_all[train_idx][:, col_indices], y_train_cls, y_train_reg,
                X_all[test_idx][:, col_indices], y_test_cls, y_test_reg,
                "full_gbm", FEATURE_GROUPS["full"],
                use_gbm=True,
            )
            res["repeat"] = repeat
            res["fold"] = fold_idx
            res["target_model"] = target_model
            res["label"] = label_config["name"]
            all_results.append(res)

        logger.info("  Repeat %d/%d complete", repeat + 1, n_repeats)

    return pd.DataFrame(all_results)


# ============================================================
# 6. Mismatched + scrambled tokenizer controls
# ============================================================

def extract_features_mismatched(
    token_lists: List[List[int]],
    matched_tokenizer,
    alt_tokenizer,
    label: str = "mismatched",
) -> pd.DataFrame:
    """Extract features by decoding with matched tokenizer then
    re-tokenizing with a different tokenizer."""
    texts = [matched_tokenizer.decode(toks[:PREFIX_LEN], skip_special_tokens=True)
             for toks in token_lists]

    merge_map = _build_merge_rank_map(alt_tokenizer)

    vocab_size = alt_tokenizer.vocab_size
    max_merge = max(merge_map.values()) if merge_map else 1
    rows = []

    for i, text in enumerate(texts):
        row: Dict = {}
        n_chars = len(text)
        row["len_chars"] = n_chars
        row["char_entropy"] = _char_entropy(text)
        row["zlib_bpc"] = _zlib_bpc(text)

        token_ids = alt_tokenizer.encode(text, add_special_tokens=False)
        n_tokens = len(token_ids)
        row["n_tokens"] = n_tokens
        row["n_special_tokens"] = 0
        row["compression_ratio"] = n_tokens / n_chars if n_chars > 0 else float("nan")
        row["log_token_count"] = math.log(n_tokens) if n_tokens > 0 else float("nan")

        if n_tokens > 0:
            if merge_map:
                ranks = [merge_map.get(tid, max_merge) for tid in token_ids]
                normed = [r / max_merge for r in ranks]
                row["merge_rank_mean"] = float(np.mean(normed))
                row["merge_rank_min"] = float(np.min(normed))
                row["merge_rank_max"] = float(np.max(normed))
            else:
                rm, rmn, rmx = _norm_id_rank_stats(token_ids, vocab_size)
                row["merge_rank_mean"] = rm
                row["merge_rank_min"] = rmn
                row["merge_rank_max"] = rmx
        else:
            row["merge_rank_mean"] = float("nan")
            row["merge_rank_min"] = float("nan")
            row["merge_rank_max"] = float("nan")

        if n_tokens > 0:
            pieces = alt_tokenizer.convert_ids_to_tokens(token_ids)
            piece_lens = [len(p) for p in pieces]
            row["max_token_len"] = max(piece_lens)
            row["mean_token_len"] = float(np.mean(piece_lens))
            row["frac_single_char_tokens"] = sum(1 for pl in piece_lens if pl == 1) / n_tokens
        else:
            for col in ["max_token_len", "mean_token_len", "frac_single_char_tokens"]:
                row[col] = float("nan")

        rows.append(row)
        if (i + 1) % 50000 == 0:
            logger.info("[%s] Features: %d / %d", label, i + 1, len(texts))

    logger.info("[%s] Feature extraction complete: %d spans", label, len(rows))
    return pd.DataFrame(rows)


def make_scrambled_tokenizer(tokenizer):
    """Create a copy of the tokenizer with randomly permuted vocabulary IDs.

    Preserves all token statistics (lengths, counts) but breaks
    any correlation between ID ordering and model internals.
    """
    scrambled = copy.deepcopy(tokenizer)
    vocab = scrambled.get_vocab()
    ids = list(vocab.values())
    rng = np.random.RandomState(12345)
    shuffled_ids = ids.copy()
    rng.shuffle(shuffled_ids)
    id_map = dict(zip(ids, shuffled_ids))

    class ScrambledWrapper:
        """Thin wrapper that permutes token IDs after encoding."""
        def __init__(self, base_tok, mapping):
            self._base = base_tok
            self._map = mapping
            self._inv = {v: k for k, v in mapping.items()}
            self.vocab_size = base_tok.vocab_size
            self.unk_token_id = base_tok.unk_token_id
            self.eos_token_id = base_tok.eos_token_id
            self.bos_token_id = base_tok.bos_token_id
            self.pad_token_id = base_tok.pad_token_id

        def encode(self, text, **kwargs):
            ids = self._base.encode(text, **kwargs)
            return [self._map.get(i, i) for i in ids]

        def decode(self, ids, **kwargs):
            orig = [self._inv.get(i, i) for i in ids]
            return self._base.decode(orig, **kwargs)

        def convert_ids_to_tokens(self, ids):
            orig = [self._inv.get(i, i) for i in ids]
            return self._base.convert_ids_to_tokens(orig)

        def convert_tokens_to_ids(self, tok):
            return self._map.get(self._base.convert_tokens_to_ids(tok))

        def get_vocab(self):
            return {k: self._map.get(v, v) for k, v in self._base.get_vocab().items()}

    return ScrambledWrapper(scrambled, id_map)


# ============================================================
# 7. SHAP analysis
# ============================================================

def run_shap_analysis(
    features_df: pd.DataFrame,
    memorization_scores: np.ndarray,
    threshold_pct: float = 5.0,
    seed: int = 42,
    max_samples: int = 5000,
) -> Dict:
    """Compute SHAP values for the full logistic model."""
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed — skipping")
        return {"error": "shap not installed"}

    X = features_df[ALL_FEATURE_COLS].values
    y = make_binary_label(memorization_scores, threshold_pct=threshold_pct)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(impute_nan_median_columns(X))

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    clf.fit(X_scaled, y)

    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X_scaled), min(max_samples, len(X_scaled)), replace=False)
    X_bg = X_scaled[idx]

    explainer = shap.LinearExplainer(clf, X_bg)
    shap_values = explainer.shap_values(X_bg)
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = dict(sorted(
        zip(ALL_FEATURE_COLS, mean_abs.tolist()), key=lambda x: -x[1]
    ))

    coef_importance = dict(sorted(
        zip(ALL_FEATURE_COLS, np.abs(clf.coef_[0]).tolist()), key=lambda x: -x[1]
    ))

    return {"shap_importance": importance, "coef_importance": coef_importance,
            "n_samples": len(X_bg)}


# ============================================================
# 8. VIF + correlation diagnostics
# ============================================================

def compute_vif(features_df: pd.DataFrame) -> pd.DataFrame:
    """Variance inflation factors for all features."""
    X = impute_nan_median_columns(features_df[ALL_FEATURE_COLS].values)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    vifs = []
    for j in range(X.shape[1]):
        y_j = X[:, j]
        X_rest = np.delete(X, j, axis=1)
        lr = LinearRegression().fit(X_rest, y_j)
        r2 = lr.score(X_rest, y_j)
        vif = 1 / (1 - r2) if r2 < 1.0 else float("inf")
        vifs.append({"feature": ALL_FEATURE_COLS[j], "VIF": vif})
    return pd.DataFrame(vifs)


def compute_feature_correlations(features_df: pd.DataFrame) -> pd.DataFrame:
    return features_df[ALL_FEATURE_COLS].corr()


def compute_partial_correlations(
    features_df: pd.DataFrame,
    memorization_scores: np.ndarray,
) -> pd.DataFrame:
    """Residualized Spearman correlation controlling for len_chars."""
    results = []
    y = memorization_scores

    for feat in ALL_FEATURE_COLS:
        x = features_df[feat].values
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() < 10:
            continue

        rho_raw, p_raw = spearmanr(x[mask], y[mask])

        length = features_df["len_chars"].values
        mask2 = mask & ~np.isnan(length)
        if mask2.sum() < 10 or feat == "len_chars":
            rho_partial, p_partial = float("nan"), float("nan")
        else:
            lr = LinearRegression()
            L = length[mask2].reshape(-1, 1)
            x_resid = x[mask2] - lr.fit(L, x[mask2]).predict(L)
            y_resid = y[mask2] - lr.fit(L, y[mask2]).predict(L)
            rho_partial, p_partial = spearmanr(x_resid, y_resid)

        results.append({
            "feature": feat,
            "spearman_raw": rho_raw, "p_raw": p_raw,
            "spearman_resid_ctrl_len": rho_partial, "p_resid": p_partial,
        })

    return pd.DataFrame(results)


# ============================================================
# 9. Summary
# ============================================================

def summarize_cv_results(cv_results: pd.DataFrame) -> pd.DataFrame:
    agg = cv_results.groupby(
        ["target_model", "label", "group", "model"]
    ).agg(
        auroc_mean=("auroc", "mean"), auroc_std=("auroc", "std"),
        auprc_mean=("auprc", "mean"), auprc_std=("auprc", "std"),
        spearman_mean=("spearman_rho", "mean"), spearman_std=("spearman_rho", "std"),
        n_folds=("auroc", "count"),
    ).reset_index()

    for metric in ["auroc", "auprc", "spearman"]:
        agg[f"{metric}_ci"] = agg.apply(
            lambda r: f"{r[f'{metric}_mean']:.3f} ± {r[f'{metric}_std']:.3f}", axis=1
        )
    return agg


# ============================================================
# 10. Main pipeline
# ============================================================

def run_experiment(
    n_samples: int = 100000,
    target_models: Optional[List[str]] = None,
    threshold_pcts: Optional[List[float]] = None,
    n_folds: int = 5,
    n_repeats: int = 3,
    seed: int = 42,
    output_dir: str = "results_v2",
    run_mismatched: bool = True,
    run_scrambled: bool = True,
    run_shap: bool = True,
    cache_dir: Optional[str] = None,
):
    if target_models is None:
        target_models = ["1.4B"]
    if threshold_pcts is None:
        threshold_pcts = [5.0, 1.0]

    os.makedirs(output_dir, exist_ok=True)
    label_configs = get_label_configs(threshold_pcts)

    # --- 1. Load ---
    df = load_memorization_dataset(n_samples=n_samples, seed=seed, cache_dir=cache_dir)
    token_lists = df["Tokens"].tolist()

    # --- 2. Features (matched, prefix-only) ---
    from transformers import AutoTokenizer
    logger.info("Loading GPT-NeoX tokenizer (matched) ...")
    matched_tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b")
    merge_map = _build_merge_rank_map(matched_tok)
    logger.info("Merge-rank map: %d entries", len(merge_map) if merge_map else 0)

    feat_path = os.path.join(output_dir, "features_matched.parquet")
    if os.path.exists(feat_path):
        features_df = pd.read_parquet(feat_path)
        logger.info("Loaded cached features (%d rows)", len(features_df))
    else:
        features_df = extract_features(token_lists, matched_tok,
                                       prefix_only=True, merge_rank_map=merge_map)
        features_df.to_parquet(feat_path, index=False)

    # --- 3. Label diagnostics ---
    for tm in target_models:
        if tm not in df.columns:
            continue
        s = df[tm].dropna()
        diag = {
            "model": tm, "n": len(s), "mean": float(s.mean()),
            "median": float(s.median()), "frac_zero": float((s == 0).mean()),
            "frac_exact_1": float((s == 1.0).mean()),
            "frac_ge_0.5": float((s >= 0.5).mean()),
            "p99": float(s.quantile(0.99)),
        }
        logger.info("Label diagnostics %s: %s", tm, diag)
        with open(os.path.join(output_dir, f"label_diagnostics_{tm}.json"), "w") as f:
            json.dump(diag, f, indent=2)

    # --- 4. CV ablation ---
    all_cv = []
    for target_model in target_models:
        if target_model not in df.columns:
            continue
        scores = df[target_model].values.astype(float)
        valid = ~np.isnan(scores)

        for lc in label_configs:
            cv_res = run_cv_ablation(
                features_df[valid].reset_index(drop=True), scores[valid],
                target_model, lc, n_folds, n_repeats, seed,
            )
            if len(cv_res):
                all_cv.append(cv_res)

    if all_cv:
        cv_all = pd.concat(all_cv, ignore_index=True)
        cv_all.to_parquet(os.path.join(output_dir, "cv_ablation_results.parquet"), index=False)
        summary = summarize_cv_results(cv_all)
        summary.to_csv(os.path.join(output_dir, "cv_ablation_summary.csv"), index=False)
        logger.info("CV summary:\n%s", summary[
            ["target_model", "label", "group", "model", "auroc_ci", "spearman_ci"]
        ].to_string(index=False))

    # --- 5. Mismatched tokenizer (GPT-2) ---
    if run_mismatched:
        logger.info("Mismatched tokenizer control (GPT-2) ...")
        mis_tok = AutoTokenizer.from_pretrained("gpt2")
        mis_path = os.path.join(output_dir, "features_mismatched_gpt2.parquet")
        if os.path.exists(mis_path):
            mis_feat = pd.read_parquet(mis_path)
        else:
            mis_feat = extract_features_mismatched(token_lists, matched_tok, mis_tok, "gpt2")
            mis_feat.to_parquet(mis_path, index=False)

        mis_cv_all = []
        for tm in target_models:
            if tm not in df.columns:
                continue
            scores = df[tm].values.astype(float)
            valid = ~np.isnan(scores)
            for lc in label_configs:
                cv_res = run_cv_ablation(
                    mis_feat[valid].reset_index(drop=True), scores[valid],
                    f"{tm}_gpt2", lc, n_folds, n_repeats, seed,
                    include_random_null=False,
                )
                if len(cv_res):
                    mis_cv_all.append(cv_res)
        if mis_cv_all:
            mis_cv = pd.concat(mis_cv_all, ignore_index=True)
            mis_cv.to_parquet(os.path.join(output_dir, "cv_mismatched_gpt2.parquet"), index=False)
            summarize_cv_results(mis_cv).to_csv(
                os.path.join(output_dir, "cv_mismatched_gpt2_summary.csv"), index=False)

    # --- 6. Scrambled tokenizer ---
    if run_scrambled:
        logger.info("Scrambled tokenizer control ...")
        scr_tok = make_scrambled_tokenizer(matched_tok)
        scr_path = os.path.join(output_dir, "features_scrambled.parquet")
        if os.path.exists(scr_path):
            scr_feat = pd.read_parquet(scr_path)
        else:
            scr_feat = extract_features_mismatched(token_lists, matched_tok, scr_tok, "scrambled")
            scr_feat.to_parquet(scr_path, index=False)

        scr_cv_all = []
        for tm in target_models:
            if tm not in df.columns:
                continue
            scores = df[tm].values.astype(float)
            valid = ~np.isnan(scores)
            for lc in label_configs:
                cv_res = run_cv_ablation(
                    scr_feat[valid].reset_index(drop=True), scores[valid],
                    f"{tm}_scrambled", lc, n_folds, n_repeats, seed,
                    include_random_null=False,
                )
                if len(cv_res):
                    scr_cv_all.append(cv_res)
        if scr_cv_all:
            scr_cv = pd.concat(scr_cv_all, ignore_index=True)
            scr_cv.to_parquet(os.path.join(output_dir, "cv_scrambled.parquet"), index=False)
            summarize_cv_results(scr_cv).to_csv(
                os.path.join(output_dir, "cv_scrambled_summary.csv"), index=False)

    # --- 7. SHAP ---
    if run_shap:
        for tm in target_models:
            if tm not in df.columns:
                continue
            scores = df[tm].values.astype(float)
            valid = ~np.isnan(scores)
            for pct in threshold_pcts:
                res = run_shap_analysis(features_df[valid].reset_index(drop=True),
                                        scores[valid], pct, seed)
                path = os.path.join(output_dir, f"shap_{tm}_top{pct}pct.json")
                with open(path, "w") as f:
                    json.dump(res, f, indent=2)

    # --- 8. VIF + correlations ---
    vif_df = compute_vif(features_df)
    vif_df.to_csv(os.path.join(output_dir, "vif_analysis.csv"), index=False)
    logger.info("VIF:\n%s", vif_df.to_string(index=False))

    compute_feature_correlations(features_df).to_csv(
        os.path.join(output_dir, "feature_correlations.csv"))

    for tm in target_models:
        if tm not in df.columns:
            continue
        scores = df[tm].values.astype(float)
        valid = ~np.isnan(scores)
        pc = compute_partial_correlations(features_df[valid].reset_index(drop=True), scores[valid])
        pc.to_csv(os.path.join(output_dir, f"partial_correlations_{tm}.csv"), index=False)

    # --- 9. Scale curve ---
    logger.info("Scale curve ...")
    scale = []
    for ms in MODEL_SIZES:
        if ms not in df.columns:
            continue
        scores = df[ms].values.astype(float)
        valid = ~np.isnan(scores)
        if valid.sum() < 100:
            continue
        y_cls = make_binary_label(scores[valid], threshold_pct=5.0)
        X = features_df[valid].reset_index(drop=True)[ALL_FEATURE_COLS].values
        X = StandardScaler().fit_transform(impute_nan_median_columns(X))
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        yp = cross_val_predict(clf, X, y_cls, cv=skf, method="predict_proba")[:, 1]
        auroc = roc_auc_score(y_cls, yp)
        scale.append({"model_size": ms, "auroc_full": auroc, "n": int(valid.sum())})
        logger.info("  %s: AUROC=%.3f", ms, auroc)
    pd.DataFrame(scale).to_csv(os.path.join(output_dir, "scale_curve.csv"), index=False)

    logger.info("Done. Results in %s/", output_dir)
    return output_dir


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Tokenizer memorization experiment v2")
    p.add_argument("--n-samples", type=int, default=100000)
    p.add_argument("--target-models", nargs="+", default=["1.4B"])
    p.add_argument("--thresholds", nargs="+", type=float, default=[5.0, 1.0])
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--n-repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="results_v2")
    p.add_argument("--no-mismatched", action="store_true")
    p.add_argument("--no-scrambled", action="store_true")
    p.add_argument("--no-shap", action="store_true")
    p.add_argument("--cache-dir", type=str, default=None)
    args = p.parse_args()

    run_experiment(
        n_samples=args.n_samples,
        target_models=args.target_models,
        threshold_pcts=args.thresholds,
        n_folds=args.n_folds,
        n_repeats=args.n_repeats,
        seed=args.seed,
        output_dir=args.output_dir,
        run_mismatched=not args.no_mismatched,
        run_scrambled=not args.no_scrambled,
        run_shap=not args.no_shap,
        cache_dir=args.cache_dir,
    )
