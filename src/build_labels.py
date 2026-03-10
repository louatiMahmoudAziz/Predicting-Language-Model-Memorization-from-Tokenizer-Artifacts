"""
build_labels.py
---------------
Build ΔBPC memorization-risk labels by joining ref and target BPC scores.

Implements SPEC.md §3.1:
  L^Tok(s) = ΔBPC^Tok(s) = BPC_{M_ref^Tok}(s) - BPC_{M_target^Tok}(s)

  ΔBPC > 0  →  target assigns higher probability per character  →  memorization signal.

Pipeline position:
  score_bpc (ref)  ─┐
                    ├──► build_labels ──► labels/<run_id>_labels.parquet
  score_bpc (target)─┘                      labels/<run_id>_labels_manifest.json

Validation checks (all fail loudly — SPEC.md §10)
--------------------------------------------------
1. Required columns present in both parquet files.
2. No duplicate candidate_id within either file.
3. tok_id must match between ref and target (SPEC §3: same tokenizer required).
4. run_id must match between ref and target.
5. Candidate ID sets must be identical — any missing or extra ID raises ValueError.

Row validity policy
-------------------
  valid_ref         : from ref file's 'valid' column
  valid_target      : from target file's 'valid' column
  valid_label       : valid_ref AND valid_target
  delta_bpc         : bpc_ref - bpc_target  (NaN unless valid_label)
  invalid_reason_label : populated when valid_label=False; explains which side failed

Only valid_label=True rows participate in quantile-threshold calculation.
Invalid rows are retained in the output with NaN scores and threshold labels.

Threshold label generation
--------------------------
Three binary label columns, each computed over valid rows only:

  label_top_5pct   : top 5% by ΔBPC among valid rows
  label_top_1pct   : top 1% by ΔBPC among valid rows
  label_top_0_1pct : top 0.1% by ΔBPC among valid rows — or NaN if not resolvable

Cutoff rule:
  k = floor(frac * N_valid)           (number of positive labels)
  cutoff = np.quantile(deltas, 1-frac)
  label = (delta_bpc >= cutoff)

  Ties at the boundary: all rows ≥ cutoff are labeled positive.
  This may yield slightly more than k positives when many rows share the exact
  cutoff value.  The cutoff value is recorded in the manifest for auditability.

0.1% resolvability rule (SPEC.md §7.3)
---------------------------------------
  if floor(0.001 * N_valid) == 0  (i.e., N_valid < 1000):
      label_top_0_1pct = NaN for all rows
      manifest: resolvable_0_1pct = false
  else:
      label computed normally
      manifest: resolvable_0_1pct = true

CLI
---
  python -m src.build_labels \\
      --ref-scores   labels/run1_ref_bpc.parquet \\
      --target-scores labels/run1_target_bpc.parquet \\
      --output        labels/ \\
      --run-id        run1

  python -m src.build_labels --config configs/colab_mini.yaml
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

_NAN = float("nan")

# Columns that must exist in a score_bpc output parquet file.
_REQUIRED_SCORE_COLS = {
    "candidate_id",
    "text_raw",
    "bpc",
    "valid",
    "tok_id",
    "run_id",
    "role",
}

# Threshold fractions and their column names.
_THRESHOLDS: List[Tuple[str, float]] = [
    ("label_top_5pct",   0.05),
    ("label_top_1pct",   0.01),
    ("label_top_0_1pct", 0.001),
]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_scores(path: str, expected_role: str) -> Any:
    """Load a score_bpc parquet file, validate schema, return DataFrame."""
    import pandas as pd  # type: ignore[import-untyped]

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Scores file not found: {path!r}")

    df = pd.read_parquet(path)

    missing = _REQUIRED_SCORE_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Scores file {path!r} is missing required columns: {sorted(missing)}"
        )

    roles = df["role"].unique().tolist()
    if len(roles) != 1 or roles[0] != expected_role:
        raise ValueError(
            f"Scores file {path!r}: expected role={expected_role!r}, "
            f"found role values={roles!r}"
        )

    logger.info(
        "Loaded %d rows from %s (role=%s)", len(df), path, expected_role
    )
    return df


def _check_no_duplicates(df: Any, path: str) -> None:
    """Fail loudly if any candidate_id appears more than once."""
    counts = df["candidate_id"].value_counts()
    dupes = counts[counts > 1].index.tolist()
    if dupes:
        raise ValueError(
            f"Scores file {path!r} contains duplicate candidate_ids: "
            f"{dupes[:10]}{'...' if len(dupes) > 10 else ''}"
        )


def _check_id_alignment(
    ids_ref: Any, ids_target: Any,
) -> None:
    """Fail loudly if ref and target do not have exactly the same candidate IDs."""
    set_ref = set(ids_ref)
    set_target = set(ids_target)

    only_ref = sorted(set_ref - set_target)
    only_target = sorted(set_target - set_ref)

    if only_ref or only_target:
        msg_parts = []
        if only_ref:
            msg_parts.append(
                f"  {len(only_ref)} id(s) in ref but not target: "
                f"{only_ref[:5]}{'...' if len(only_ref) > 5 else ''}"
            )
        if only_target:
            msg_parts.append(
                f"  {len(only_target)} id(s) in target but not ref: "
                f"{only_target[:5]}{'...' if len(only_target) > 5 else ''}"
            )
        raise ValueError(
            "Ref and target score files have mismatched candidate IDs:\n"
            + "\n".join(msg_parts)
        )


# ---------------------------------------------------------------------------
# Label computation helpers
# ---------------------------------------------------------------------------

def _is_null(v: Any) -> bool:
    """Return True if a value is None, NaN, pd.NA, or pd.NaT."""
    import pandas as pd  # type: ignore[import-untyped]
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _compute_thresholds(
    delta_bpc_valid: Any,  # numpy array of valid delta_bpc values
    n_valid: int,
) -> Dict[str, Any]:
    """
    Compute cutoff value and k for each threshold fraction.

    Returns a dict keyed by column name with:
      {"cutoff": float, "k": int, "resolvable": bool, "fraction": float}
    """
    info: Dict[str, Any] = {}

    for col, frac in _THRESHOLDS:
        k = int(math.floor(frac * n_valid))
        resolvable = k >= 1

        if not resolvable:
            info[col] = {
                "fraction": frac,
                "k": 0,
                "cutoff": None,
                "resolvable": False,
            }
            logger.warning(
                "Threshold %s (top %.1f%%): not resolvable — "
                "floor(%.4f * %d) = 0 valid rows would be labeled positive. "
                "Column will be NaN for all rows.",
                col, frac * 100, frac, n_valid,
            )
        else:
            # np.quantile(arr, q) where q = 1-frac gives the value
            # above which frac of the data lies.
            cutoff = float(np.quantile(delta_bpc_valid, 1.0 - frac))
            info[col] = {
                "fraction": frac,
                "k": k,
                "cutoff": cutoff,
                "resolvable": True,
            }
            logger.info(
                "Threshold %s: top %.1f%% of %d valid rows → k=%d, cutoff=%.6f",
                col, frac * 100, n_valid, k, cutoff,
            )

    return info


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def build_labels(
    ref_scores_path: str,
    target_scores_path: str,
    output_dir: str,
    run_id: str,
) -> str:
    """
    Join ref and target BPC scores, compute ΔBPC, and generate threshold labels.

    Parameters
    ----------
    ref_scores_path    : parquet file produced by score_bpc with role="ref"
    target_scores_path : parquet file produced by score_bpc with role="target"
    output_dir         : directory for output parquet + manifest
    run_id             : experiment run identifier

    Returns
    -------
    str  Path to the written labels parquet file.
    """
    import pandas as pd  # type: ignore[import-untyped]

    logger.info("=" * 65)
    logger.info("build_labels: run_id=%s", run_id)
    logger.info("=" * 65)

    # ------------------------------------------------------------------
    # Load and validate
    # ------------------------------------------------------------------
    df_ref = _load_scores(ref_scores_path, expected_role="ref")
    df_target = _load_scores(target_scores_path, expected_role="target")

    _check_no_duplicates(df_ref, ref_scores_path)
    _check_no_duplicates(df_target, target_scores_path)

    # tok_id must match
    tok_ids_ref = df_ref["tok_id"].unique().tolist()
    tok_ids_target = df_target["tok_id"].unique().tolist()
    if len(tok_ids_ref) != 1:
        raise ValueError(
            f"Ref scores file has multiple tok_id values: {tok_ids_ref}. "
            f"Each scores file must correspond to exactly one tokenizer."
        )
    if len(tok_ids_target) != 1:
        raise ValueError(
            f"Target scores file has multiple tok_id values: {tok_ids_target}. "
            f"Each scores file must correspond to exactly one tokenizer."
        )
    tok_id = tok_ids_ref[0]
    if tok_ids_ref[0] != tok_ids_target[0]:
        raise ValueError(
            f"Tokenizer mismatch: ref tok_id={tok_ids_ref[0]!r} but "
            f"target tok_id={tok_ids_target[0]!r}. "
            f"Per SPEC.md §3, ref and target must share the same tokenizer."
        )

    # run_id must match
    run_ids_ref = df_ref["run_id"].unique().tolist()
    run_ids_target = df_target["run_id"].unique().tolist()
    if run_ids_ref != run_ids_target:
        raise ValueError(
            f"run_id mismatch: ref={run_ids_ref!r}, target={run_ids_target!r}. "
            f"Both score files must belong to the same experimental run."
        )

    # candidate_id alignment
    _check_id_alignment(df_ref["candidate_id"], df_target["candidate_id"])

    # ------------------------------------------------------------------
    # Merge on candidate_id, preserving ref order
    # ------------------------------------------------------------------
    # Rename columns to avoid conflicts before merge
    ref_rename = {
        "bpc": "bpc_ref",
        "total_bits": "total_bits_ref",
        "valid": "valid_ref",
        "invalid_reason": "invalid_reason_ref",
        "n_tokens": "n_tokens_ref",
        "len_chars": "len_chars_ref",
        "normalizer_id": "normalizer_id_ref",
        "truncated": "truncated_ref",
        "scored_at": "scored_at_ref",
        "model_dir": "model_dir_ref",
        "normalized_text": "normalized_text_ref",
    }
    target_rename = {
        "bpc": "bpc_target",
        "total_bits": "total_bits_target",
        "valid": "valid_target",
        "invalid_reason": "invalid_reason_target",
        "n_tokens": "n_tokens_target",
        "len_chars": "len_chars_target",
        "normalizer_id": "normalizer_id_target",
        "truncated": "truncated_target",
        "scored_at": "scored_at_target",
        "model_dir": "model_dir_target",
        "normalized_text": "normalized_text_target",
    }

    # Only rename columns that exist in the frame (graceful if extras absent)
    ref_rename = {k: v for k, v in ref_rename.items() if k in df_ref.columns}
    target_rename = {k: v for k, v in target_rename.items() if k in df_target.columns}

    df_ref = df_ref.rename(columns=ref_rename)
    df_target = df_target.rename(columns=target_rename)

    # Keep only candidate_id (join key) + renamed target columns.
    # text_raw, tok_id, run_id are already validated to match and come from df_ref.
    target_keep = {"candidate_id"} | set(target_rename.values())
    df_target_slim = df_target[[c for c in df_target.columns if c in target_keep]]

    df = df_ref.merge(df_target_slim, on="candidate_id", how="inner")

    # Sanity: inner join should preserve all rows since ID sets are identical
    if len(df) != len(df_ref):
        raise RuntimeError(
            f"Merge produced {len(df)} rows but expected {len(df_ref)}. "
            "This is a bug — please report it."
        )

    # Drop columns that are noise in a labels file (role is always "ref"
    # from the ref side; it's misleading in a joined labels context).
    for drop_col in ["role"]:
        if drop_col in df.columns:
            df = df.drop(columns=[drop_col])

    # ------------------------------------------------------------------
    # Compute valid_label and delta_bpc
    # ------------------------------------------------------------------
    df["valid_ref"] = df["valid_ref"].astype(bool)
    df["valid_target"] = df["valid_target"].astype(bool)
    df["valid_label"] = df["valid_ref"] & df["valid_target"]

    # delta_bpc only for rows where both sides are valid
    df["delta_bpc"] = _NAN
    valid_mask = df["valid_label"]
    df.loc[valid_mask, "delta_bpc"] = (
        df.loc[valid_mask, "bpc_ref"].values
        - df.loc[valid_mask, "bpc_target"].values
    )

    # Build invalid_reason_label from ref/target reasons
    def _build_reason(row: Any) -> Optional[str]:
        if row["valid_label"]:
            return None
        parts = []
        rr = row.get("invalid_reason_ref")
        tr = row.get("invalid_reason_target")
        if not _is_null(rr):
            parts.append(f"ref:{rr}")
        if not _is_null(tr):
            parts.append(f"target:{tr}")
        return " | ".join(parts) if parts else "invalid_score"

    df["invalid_reason_label"] = df.apply(_build_reason, axis=1)

    # ------------------------------------------------------------------
    # Threshold label generation
    # ------------------------------------------------------------------
    valid_deltas = df.loc[valid_mask, "delta_bpc"].values.astype(float)
    n_valid = int(valid_mask.sum())

    n_nan_in_valid = int(np.isnan(valid_deltas).sum())
    if n_nan_in_valid > 0:
        raise ValueError(
            f"{n_nan_in_valid} rows have valid_label=True but delta_bpc=NaN. "
            f"This means bpc_ref or bpc_target is NaN despite valid=True — "
            f"likely a bug in score_bpc.py.  Fix upstream before building labels."
        )

    logger.info(
        "Label statistics: %d total rows, %d valid_label, %d invalid",
        len(df), n_valid, len(df) - n_valid,
    )

    if n_valid == 0:
        logger.warning(
            "No valid rows for threshold label generation. "
            "All threshold columns will be NaN."
        )
        threshold_info: Dict[str, Any] = {
            col: {"fraction": frac, "k": 0, "cutoff": None, "resolvable": False}
            for col, frac in _THRESHOLDS
        }
        for col, _ in _THRESHOLDS:
            df[col] = _NAN
    else:
        threshold_info = _compute_thresholds(valid_deltas, n_valid)

        for col, frac in _THRESHOLDS:
            info = threshold_info[col]
            if not info["resolvable"]:
                df[col] = _NAN
            else:
                cutoff = info["cutoff"]
                # Initialize all to NaN (includes invalid rows)
                labels = [_NAN] * len(df)
                for i, (is_valid, delta) in enumerate(
                    zip(df["valid_label"].values, df["delta_bpc"].values)
                ):
                    if is_valid:
                        labels[i] = float(delta >= cutoff)
                df[col] = labels

    # ------------------------------------------------------------------
    # Column ordering
    # ------------------------------------------------------------------
    # Core columns first, then threshold labels, then diagnostics
    core_cols = [
        "candidate_id",
        "text_raw",
        "bpc_ref",
        "bpc_target",
        "delta_bpc",
        "valid_ref",
        "valid_target",
        "valid_label",
        "invalid_reason_ref",
        "invalid_reason_target",
        "invalid_reason_label",
        "label_top_5pct",
        "label_top_1pct",
        "label_top_0_1pct",
        "tok_id",
        "run_id",
    ]
    extra_cols = [c for c in df.columns if c not in core_cols]
    df = df[core_cols + extra_cols]

    # ------------------------------------------------------------------
    # Write parquet
    # ------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{run_id}_labels.parquet")
    df.to_parquet(out_path, index=False, engine="pyarrow")
    logger.info("Labels written -> %s", out_path)

    # ------------------------------------------------------------------
    # Write manifest
    # ------------------------------------------------------------------
    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "tok_id": tok_id,
        "built_at": datetime.datetime.utcnow().isoformat() + "Z",
        "inputs": {
            "ref_scores": os.path.abspath(ref_scores_path),
            "target_scores": os.path.abspath(target_scores_path),
        },
        "row_counts": {
            "total": len(df),
            "valid_label": n_valid,
            "invalid_ref_only": int((~df["valid_ref"] & df["valid_target"]).sum()),
            "invalid_target_only": int((df["valid_ref"] & ~df["valid_target"]).sum()),
            "invalid_both": int((~df["valid_ref"] & ~df["valid_target"]).sum()),
        },
        "delta_bpc_stats": (
            {
                "mean": float(np.mean(valid_deltas)),
                "std": float(np.std(valid_deltas)),
                "min": float(np.min(valid_deltas)),
                "max": float(np.max(valid_deltas)),
                "p25": float(np.quantile(valid_deltas, 0.25)),
                "p50": float(np.quantile(valid_deltas, 0.50)),
                "p75": float(np.quantile(valid_deltas, 0.75)),
                "p99": float(np.quantile(valid_deltas, 0.99)),
            }
            if n_valid > 0
            else None
        ),
        "thresholds": {
            col: {
                "fraction": info["fraction"],
                "k_expected": info["k"],
                "cutoff": info["cutoff"],
                "resolvable": info["resolvable"],
                "k_actual": (
                    int((df[col] == 1.0).sum())
                    if info["resolvable"]
                    else 0
                ),
            }
            for col, info in threshold_info.items()
        },
    }

    manifest_path = os.path.join(output_dir, f"{run_id}_labels_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info("Manifest written -> %s", manifest_path)

    # Summary
    for col, info in threshold_info.items():
        if info["resolvable"]:
            k_actual = int((df[col] == 1.0).sum())
            logger.info(
                "  %-22s  cutoff=%.6f  k_expected=%d  k_actual=%d",
                col, info["cutoff"], info["k"], k_actual,
            )
        else:
            logger.info("  %-22s  NOT RESOLVABLE (n_valid=%d)", col, n_valid)

    logger.info("=" * 65)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.build_labels",
        description="Build ΔBPC memorization-risk labels from ref/target BPC scores.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", metavar="YAML", help="Project YAML config.")
    p.add_argument(
        "--ref-scores", metavar="FILE",
        help="Parquet from score_bpc with role=ref.",
    )
    p.add_argument(
        "--target-scores", metavar="FILE",
        help="Parquet from score_bpc with role=target.",
    )
    p.add_argument("--output", metavar="DIR", help="Output directory.")
    p.add_argument("--run-id", metavar="ID", help="Run identifier.")
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

    ref_scores = target_scores = output_dir = run_id = None

    if args.config:
        from src.config import load_config  # type: ignore[import]
        cfg = load_config(args.config)
        run_id = cfg.run_id
        output_dir = cfg.paths.labels_dir
        # Derive default score file paths from config conventions
        ref_scores = os.path.join(cfg.paths.labels_dir, f"{run_id}_ref_bpc.parquet")
        target_scores = os.path.join(cfg.paths.labels_dir, f"{run_id}_target_bpc.parquet")

    if args.ref_scores:     ref_scores = args.ref_scores
    if args.target_scores:  target_scores = args.target_scores
    if args.output:         output_dir = args.output
    if args.run_id:         run_id = args.run_id

    missing = [
        name for name, val in [
            ("--ref-scores",    ref_scores),
            ("--target-scores", target_scores),
            ("--output",        output_dir),
            ("--run-id",        run_id),
        ]
        if val is None
    ]
    if missing:
        parser.error("Missing required arguments:\n  " + "\n  ".join(missing))

    try:
        out = build_labels(
            ref_scores_path=ref_scores,          # type: ignore[arg-type]
            target_scores_path=target_scores,    # type: ignore[arg-type]
            output_dir=output_dir,               # type: ignore[arg-type]
            run_id=run_id,                       # type: ignore[arg-type]
        )
    except (FileNotFoundError, ValueError, RuntimeError, ImportError) as e:
        logger.error("FATAL: %s", e)
        sys.exit(1)

    print(f"\nDone.  Labels written to: {out}")


if __name__ == "__main__":
    main()
