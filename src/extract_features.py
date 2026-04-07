"""
extract_features.py
-------------------
Compute tokenizer-only features for every candidate string and write one
Parquet row per candidate.

Feature categories (SPEC.md §6)
--------------------------------
1. Baseline (target-normalized string):
     len_chars, char_entropy, n_tokens_target

2. Tokenizer artifact features (target tokenizer only):
     Token rank percentiles (mean/min/max) — both families
     BPE merge-rank stats   (mean/max)     — BPE only,    null for unigram
     Unigram piece scores   (mean/min/max) — unigram only, null for BPE

3. Compression discrepancy (per reference tokenizer):
     n_tokens_ref, delta_tok = n_tokens_ref - n_tokens_target
     Uses each ref tokenizer's OWN normalizer (may differ from target)

Scientific invariants
---------------------
- Target features always use target-normalized s'.
- Reference token counts always use reference-normalized s'' (may differ from s').
- BPE and unigram artifact features are NEVER merged or treated as equivalent.
- Features unavailable for a family are stored as NaN (float) or None (object),
  with an explicit sentinel column name (e.g. merge_rank_mean = NaN for unigram).
- zlib compression baseline: zlib_bpc provides a model-free memorability proxy.
- No label-derived features. No model weights.

CLI
---
  python -m src.extract_features --target-meta tokenizers/mini_bpe/metadata.json \\
      --ref-meta tokenizers/hpc_unigram/metadata.json \\
      --candidates data/candidates/canaries_mini.jsonl \\
      --run-id colab_mini --output features/

  python -m src.extract_features --config configs/colab_mini.yaml [overrides]
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

import zlib

import numpy as np

logger = logging.getLogger(__name__)

# NaN sentinel used for features unavailable for a tokenizer family
_NAN = float("nan")


# ---------------------------------------------------------------------------
# zlib baseline — model-free compression features
# ---------------------------------------------------------------------------

def _zlib_bpc(text: str) -> float:
    """Bits-per-character via zlib compression (model-free baseline)."""
    if not text:
        return _NAN
    encoded = text.encode("utf-8")
    compressed = zlib.compress(encoded, level=9)
    return (len(compressed) * 8.0) / len(text)


def _compression_ratio(text: str) -> float:
    """Ratio of compressed size to original size (lower = more compressible)."""
    if not text:
        return _NAN
    encoded = text.encode("utf-8")
    compressed = zlib.compress(encoded, level=9)
    return len(compressed) / len(encoded)


# ---------------------------------------------------------------------------
# Tokenizer loader — dispatches on metadata["family"]
# ---------------------------------------------------------------------------

class _BPEHandle:
    """Loaded BPE tokenizer with pre-built rank and merge-rank tables."""

    def __init__(self, meta: Dict, tok_obj: Any, vocab: Dict[str, int],
                 merge_ranks: Dict[Tuple[str, str], int]) -> None:
        self.meta = meta
        self.tok_obj = tok_obj              # tokenizers.Tokenizer
        self.vocab = vocab                  # piece -> id (== rank)
        self.merge_ranks = merge_ranks      # (a, b) -> rank (0 = first merge)
        self.vocab_size = meta["vocab_size_actual"]
        self.norm_fn: Callable[[str], str] = _make_hf_norm_fn(tok_obj)
        self.normalizer_id: str = meta["normalizer_id"]

    def encode(self, s_norm: str) -> Tuple[List[int], List[str]]:
        """Encode once; return (ids, piece_strings).

        Standard Unicode normalizers (NFC/NFD/NFKC/NFKD, lowercase) are
        idempotent, so passing an already-normalized string through the
        tokenizer's internal normalizer produces the same result.
        """
        enc = self.tok_obj.encode(s_norm)
        return enc.ids, enc.tokens

    def encode_ids(self, s_norm: str) -> List[int]:
        return self.tok_obj.encode(s_norm).ids

    def token_rank_percentiles(self, ids: List[int]) -> List[float]:
        """Convert token ids to rank percentiles in [0, 1]."""
        denom = max(self.vocab_size - 1, 1)
        return [i / denom for i in ids]

    def merge_ranks_for_pieces(self, pieces: List[str]) -> List[int]:
        """Return merge ranks for pre-computed token pieces.

        Accepts the piece strings from a previous encode() call so the
        string is not redundantly re-encoded.  Single-character pieces and
        special tokens have no merge rank and are excluded.
        """
        if not self.merge_ranks:
            return []
        ranks = []
        for piece in pieces:
            rank = self._piece_merge_rank(piece)
            if rank is not None:
                ranks.append(rank)
        return ranks

    def _piece_merge_rank(self, piece: str) -> Optional[int]:
        """Return the merge rank of the merge that produced `piece`, or None.

        In standard BPE, each piece is produced by exactly one merge rule,
        so at most one split point matches.  We scan all split points
        defensively in case of non-standard merge tables, but in practice
        exactly one (or zero, for base-character leaves) will match.
        Returns None for single-character pieces (base tokens with no merge).
        """
        best = None
        n = len(piece)
        for split in range(1, n):
            a, b = piece[:split], piece[split:]
            rank = self.merge_ranks.get((a, b))
            if rank is not None:
                if best is None or rank > best:
                    best = rank
        return best


class _UnigramHandle:
    """Loaded SentencePiece Unigram tokenizer with piece-score table."""

    def __init__(self, meta: Dict, sp_obj: Any,
                 piece_scores: Dict[str, float]) -> None:
        self.meta = meta
        self.sp_obj = sp_obj                # spm.SentencePieceProcessor
        self.piece_scores = piece_scores    # piece -> log-prob score
        self.vocab_size = meta["vocab_size_actual"]
        self.norm_fn: Callable[[str], str] = _make_sp_norm_fn(sp_obj)
        self.normalizer_id: str = meta["normalizer_id"]

    def encode_ids(self, s_norm: str) -> List[int]:
        """Encode a pre-normalized string.

        SentencePiece.encode() applies its normalizer internally.  For
        standard normalizers (NFKC) this is idempotent on already-normalized
        input, so the token sequence is consistent with Normalize_Tok(s).
        """
        return self.sp_obj.encode(s_norm)

    def token_rank_percentiles(self, ids: List[int]) -> List[float]:
        denom = max(self.vocab_size - 1, 1)
        return [i / denom for i in ids]

    def piece_scores_for_tokens(self, ids: List[int]) -> List[float]:
        """Return log-prob piece scores for the given token ids.

        Pieces not found in the .vocab lookup (e.g., byte-fallback pieces)
        are excluded from the returned list with a logged warning, so
        downstream aggregation reflects only scoreable pieces.
        """
        scores = []
        n_missing = 0
        for i in ids:
            piece = self.sp_obj.id_to_piece(i)
            score = self.piece_scores.get(piece)
            if score is not None:
                scores.append(score)
            else:
                n_missing += 1
        if n_missing:
            logger.warning(
                "Unigram tok_id=%s: %d / %d token pieces not found in .vocab "
                "file; their scores are excluded from aggregation.",
                self.meta.get("tok_id"), n_missing, len(ids),
            )
        return scores


# ---------------------------------------------------------------------------
# Normalization function constructors
# ---------------------------------------------------------------------------

def _make_hf_norm_fn(tok_obj: Any) -> Callable[[str], str]:
    from src.normalize import get_normalizer_fn  # type: ignore[import]
    fn, _ = get_normalizer_fn(tok_obj)
    return fn


def _make_sp_norm_fn(sp_obj: Any) -> Callable[[str], str]:
    from src.normalize import get_normalizer_fn  # type: ignore[import]
    fn, _ = get_normalizer_fn(sp_obj)
    return fn


# ---------------------------------------------------------------------------
# Metadata loader — entry point for all downstream modules
# ---------------------------------------------------------------------------

def load_tokenizer_from_metadata(metadata_path: str) -> Any:
    """
    Load a trained tokenizer from its metadata.json file.

    Returns a _BPEHandle or _UnigramHandle depending on the family field.

    Raises
    ------
    FileNotFoundError   if metadata or any artifact file is missing
    ValueError          if metadata is missing required fields
    ImportError         if the required library is not installed
    """
    metadata_path = os.path.abspath(metadata_path)
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"Tokenizer metadata not found: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)

    required = {"tok_id", "family", "vocab_size_actual", "normalizer_id", "artifacts"}
    missing = required - meta.keys()
    if missing:
        raise ValueError(
            f"metadata.json at {metadata_path!r} is missing required fields: {sorted(missing)}"
        )

    family = meta["family"]
    artifacts = meta["artifacts"]

    if family == "bpe":
        return _load_bpe(meta, artifacts, metadata_path)
    elif family == "unigram":
        return _load_unigram(meta, artifacts, metadata_path)
    else:
        raise ValueError(
            f"Unknown tokenizer family {family!r} in {metadata_path!r}. "
            f"Expected 'bpe' or 'unigram'."
        )


def _require_artifact(artifacts: Dict, key: str, metadata_path: str) -> str:
    path = artifacts.get(key)
    if not path:
        raise ValueError(
            f"metadata.json at {metadata_path!r} is missing artifacts['{key}']."
        )
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Artifact file not found: {path!r} "
            f"(referenced from {metadata_path!r})"
        )
    return path


def _load_bpe(meta: Dict, artifacts: Dict, metadata_path: str) -> _BPEHandle:
    try:
        from tokenizers import Tokenizer  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "Hugging Face 'tokenizers' library required. "
            "pip install tokenizers"
        ) from e

    tok_json_path = _require_artifact(artifacts, "tokenizer_json", metadata_path)
    vocab_json_path = _require_artifact(artifacts, "vocab_json", metadata_path)

    tok_obj = Tokenizer.from_file(tok_json_path)

    with open(vocab_json_path, "r", encoding="utf-8") as fh:
        vocab: Dict[str, int] = json.load(fh)

    # Load merge ranks from merges.txt (rank = 0-indexed line number after header)
    merge_ranks: Dict[Tuple[str, str], int] = {}
    merges_path = artifacts.get("merges_txt")
    if merges_path and os.path.isfile(merges_path):
        rank = 0
        with open(merges_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line or line.startswith("#"):
                    continue
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    merge_ranks[(parts[0], parts[1])] = rank
                    rank += 1
        logger.debug("Loaded %d merge rules from %s", len(merge_ranks), merges_path)
    else:
        logger.warning(
            "merges.txt not found for BPE tokenizer %s; "
            "merge_rank features will be NaN.",
            meta.get("tok_id"),
        )

    return _BPEHandle(meta=meta, tok_obj=tok_obj, vocab=vocab, merge_ranks=merge_ranks)


def _load_unigram(meta: Dict, artifacts: Dict, metadata_path: str) -> _UnigramHandle:
    try:
        import sentencepiece as spm  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "Google 'sentencepiece' library required. "
            "pip install sentencepiece"
        ) from e

    model_path = _require_artifact(artifacts, "model", metadata_path)
    vocab_path = _require_artifact(artifacts, "vocab", metadata_path)

    sp_obj = spm.SentencePieceProcessor(model_file=model_path)

    # Load piece scores from .vocab (tab-separated: piece \t score)
    piece_scores: Dict[str, float] = {}
    with open(vocab_path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                raise ValueError(
                    f"Malformed .vocab file {vocab_path!r} at line {lineno}: "
                    f"expected 2 tab-separated columns, got {len(parts)}: {line!r}"
                )
            piece, score_str = parts
            try:
                piece_scores[piece] = float(score_str)
            except ValueError:
                raise ValueError(
                    f"Malformed .vocab file {vocab_path!r} at line {lineno}: "
                    f"score {score_str!r} is not a float."
                )

    logger.debug(
        "Loaded %d piece scores from %s", len(piece_scores), vocab_path
    )
    return _UnigramHandle(meta=meta, sp_obj=sp_obj, piece_scores=piece_scores)


# ---------------------------------------------------------------------------
# Candidate file loader
# ---------------------------------------------------------------------------

def load_candidates(path: str) -> List[Dict]:
    """
    Load candidate strings from a JSONL file.

    Each line must be a JSON object with at least "id" and "text" fields.
    Fails loudly on malformed lines or missing fields.
    Returns a list of dicts with at minimum {"id": str, "text": str}.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Candidates file not found: {path}")

    candidates = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Candidates file {path!r} line {lineno}: invalid JSON — {e}"
                )
            if not isinstance(obj, dict):
                raise ValueError(
                    f"Candidates file {path!r} line {lineno}: "
                    f"expected JSON object, got {type(obj).__name__}"
                )
            for field in ("id", "text"):
                if field not in obj:
                    raise ValueError(
                        f"Candidates file {path!r} line {lineno}: "
                        f"missing required field {field!r}"
                    )
            if not isinstance(obj["id"], str) or not obj["id"].strip():
                raise ValueError(
                    f"Candidates file {path!r} line {lineno}: "
                    f"'id' must be a non-empty string"
                )
            if not isinstance(obj["text"], str):
                raise ValueError(
                    f"Candidates file {path!r} line {lineno}: "
                    f"'text' must be a string"
                )
            candidates.append(obj)

    # Validate unique ids
    ids_seen: Dict[str, int] = {}
    for c in candidates:
        if c["id"] in ids_seen:
            raise ValueError(
                f"Duplicate candidate id {c['id']!r} in {path!r}."
            )
        ids_seen[c["id"]] = 1

    if not candidates:
        raise ValueError(f"Candidates file {path!r} contains no entries.")

    logger.info("Loaded %d candidates from %s", len(candidates), path)
    return candidates


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def _char_entropy(s: str) -> float:
    """Shannon entropy over the character distribution of s (bits)."""
    if not s:
        return _NAN
    freq: Dict[str, int] = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    n = len(s)
    return -sum((cnt / n) * math.log2(cnt / n) for cnt in freq.values())


def _safe_agg(values: List[float]) -> Tuple[float, float, float]:
    """Return (mean, min, max) or (NaN, NaN, NaN) if values is empty."""
    if not values:
        return _NAN, _NAN, _NAN
    return (
        float(np.mean(values)),
        float(np.min(values)),
        float(np.max(values)),
    )


def _compute_row(
    candidate: Dict,
    target: Any,               # _BPEHandle | _UnigramHandle
    refs: List[Any],           # list of _BPEHandle | _UnigramHandle
) -> Dict:
    """Compute one feature row for a single candidate string."""
    cid = candidate["id"]
    text_raw = candidate["text"]
    row: Dict[str, Any] = {
        "candidate_id": cid,
        "text_raw": text_raw,
        "target_tok_id": target.meta["tok_id"],
        "target_family": target.meta["family"],
        "target_normalizer_id": target.normalizer_id,
    }

    # ------------------------------------------------------------------ #
    # 1. Baseline features using TARGET normalization                      #
    # ------------------------------------------------------------------ #
    s_norm = target.norm_fn(text_raw)
    row["target_norm"] = s_norm

    if not s_norm:
        # Empty after normalization: mark all features as NaN per SPEC §2.4
        logger.warning(
            "Candidate id=%s produced empty string after target normalization; "
            "all features will be NaN.",
            cid,
        )
        row.update({
            "len_chars": 0,
            "char_entropy": _NAN,
            "zlib_bpc": _NAN,
            "zlib_compression_ratio": _NAN,
            "n_tokens_target": 0,
            "tok_rank_mean": _NAN, "tok_rank_min": _NAN, "tok_rank_max": _NAN,
            "merge_rank_mean": _NAN, "merge_rank_max": _NAN,
            "piece_score_mean": _NAN, "piece_score_min": _NAN, "piece_score_max": _NAN,
        })
        _add_ref_features(row, text_raw, refs, n_tokens_target=0)
        row["extracted_at"] = datetime.datetime.utcnow().isoformat() + "Z"
        return row

    row["len_chars"] = len(s_norm)
    row["char_entropy"] = _char_entropy(s_norm)
    row["zlib_bpc"] = _zlib_bpc(s_norm)
    row["zlib_compression_ratio"] = _compression_ratio(s_norm)

    # ------------------------------------------------------------------ #
    # Encode once.  For BPE we also capture piece strings so merge-rank   #
    # computation does not redundantly re-encode.                          #
    # ------------------------------------------------------------------ #
    pieces: Optional[List[str]] = None
    if isinstance(target, _BPEHandle):
        ids, pieces = target.encode(s_norm)
    elif isinstance(target, _UnigramHandle):
        ids = target.encode_ids(s_norm)
    else:
        raise TypeError(f"Unknown tokenizer handle type: {type(target)}")

    row["n_tokens_target"] = len(ids)

    if not ids and s_norm:
        logger.warning(
            "Candidate id=%s: tokenizer produced 0 tokens for a non-empty "
            "normalized string (len=%d).  All token-level features will be NaN.",
            cid, len(s_norm),
        )

    # ------------------------------------------------------------------ #
    # 2. Token rank features (both families)                               #
    # ------------------------------------------------------------------ #
    rank_pcts = target.token_rank_percentiles(ids)
    r_mean, r_min, r_max = _safe_agg(rank_pcts)
    row["tok_rank_mean"] = r_mean
    row["tok_rank_min"] = r_min
    row["tok_rank_max"] = r_max

    # ------------------------------------------------------------------ #
    # 2a. BPE-only: merge-rank statistics                                  #
    # Null (NaN) for unigram — stored explicitly, never omitted.           #
    # ------------------------------------------------------------------ #
    if isinstance(target, _BPEHandle):
        assert pieces is not None
        merge_ranks = target.merge_ranks_for_pieces(pieces)
        if merge_ranks:
            row["merge_rank_mean"] = float(np.mean(merge_ranks))
            row["merge_rank_max"]  = float(np.max(merge_ranks))
        else:
            row["merge_rank_mean"] = _NAN
            row["merge_rank_max"]  = _NAN
        # Unigram columns explicitly null
        row["piece_score_mean"] = _NAN
        row["piece_score_min"]  = _NAN
        row["piece_score_max"]  = _NAN

    # ------------------------------------------------------------------ #
    # 2b. Unigram-only: piece score statistics                             #
    # Null (NaN) for BPE — stored explicitly, never omitted.              #
    # ------------------------------------------------------------------ #
    elif isinstance(target, _UnigramHandle):
        scores = target.piece_scores_for_tokens(ids)
        ps_mean, ps_min, ps_max = _safe_agg(scores)
        row["piece_score_mean"] = ps_mean
        row["piece_score_min"]  = ps_min
        row["piece_score_max"]  = ps_max
        # BPE columns explicitly null
        row["merge_rank_mean"] = _NAN
        row["merge_rank_max"]  = _NAN

    # ------------------------------------------------------------------ #
    # 3. Compression discrepancy features (per reference tokenizer)        #
    # Each ref uses its OWN normalizer — explicitly separate from target.  #
    # ------------------------------------------------------------------ #
    _add_ref_features(row, text_raw, refs, n_tokens_target=len(ids))

    row["extracted_at"] = datetime.datetime.utcnow().isoformat() + "Z"
    return row


def _add_ref_features(
    row: Dict,
    text_raw: str,
    refs: List[Any],
    n_tokens_target: int,
) -> None:
    """
    Append compression-discrepancy columns for each reference tokenizer.

    Column names are prefixed with ref_<tok_id>_ to prevent any confusion
    with target-normalized quantities.  Each ref tokenizer applies its OWN
    normalizer to text_raw before encoding.
    """
    for ref in refs:
        prefix = f"ref_{ref.meta['tok_id']}"
        s_ref_norm = ref.norm_fn(text_raw)
        row[f"{prefix}_normalizer_id"] = ref.normalizer_id

        if not s_ref_norm:
            row[f"{prefix}_n_tokens"] = 0
            row[f"{prefix}_delta_tok"] = 0 - n_tokens_target
            continue

        ref_ids = ref.encode_ids(s_ref_norm)
        n_ref = len(ref_ids)
        row[f"{prefix}_n_tokens"] = n_ref
        row[f"{prefix}_delta_tok"] = n_ref - n_tokens_target


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def extract_features(
    target_metadata_path: str,
    candidates_path: str,
    output_dir: str,
    run_id: str,
    ref_metadata_paths: Optional[List[str]] = None,
) -> str:
    """
    Extract tokenizer-only features for all candidate strings.

    Parameters
    ----------
    target_metadata_path : path to the TARGET tokenizer's metadata.json
    candidates_path      : path to the JSONL candidate strings file
    output_dir           : root features directory
    run_id               : used as output filename stem
    ref_metadata_paths   : list of paths to reference tokenizer metadata.json files

    Returns
    -------
    str  Path to the written Parquet file.
    """
    try:
        import pandas as pd  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError("pandas is required. pip install pandas") from e

    try:
        import pyarrow  # type: ignore[import-untyped]  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "pyarrow is required for Parquet output. pip install pyarrow"
        ) from e

    logger.info("=== extract_features: run_id=%s ===", run_id)

    # Load target tokenizer
    logger.info("Loading target tokenizer from %s", target_metadata_path)
    target = load_tokenizer_from_metadata(target_metadata_path)
    logger.info(
        "Target: tok_id=%s  family=%s  normalizer=%s",
        target.meta["tok_id"], target.meta["family"], target.normalizer_id,
    )

    # Load reference tokenizers
    refs: List[Any] = []
    ref_ids_seen: set = set()
    for path in (ref_metadata_paths or []):
        logger.info("Loading reference tokenizer from %s", path)
        ref = load_tokenizer_from_metadata(path)
        rid = ref.meta["tok_id"]
        if rid == target.meta["tok_id"]:
            raise ValueError(
                f"Reference tokenizer tok_id {rid!r} is the same "
                f"as the target tokenizer.  Use a different tok_id for each."
            )
        if rid in ref_ids_seen:
            raise ValueError(
                f"Duplicate reference tokenizer tok_id {rid!r}.  "
                f"Each reference tokenizer must have a unique tok_id "
                f"to avoid column name collisions in the output Parquet."
            )
        ref_ids_seen.add(rid)
        refs.append(ref)
        logger.info(
            "Ref: tok_id=%s  family=%s  normalizer=%s",
            rid, ref.meta["family"], ref.normalizer_id,
        )

    # Load candidates
    candidates = load_candidates(candidates_path)
    n = len(candidates)

    # Compute features row by row
    rows = []
    for i, cand in enumerate(candidates):
        if (i + 1) % 500 == 0 or (i + 1) == n:
            logger.info("Processing %d / %d", i + 1, n)
        try:
            row = _compute_row(cand, target, refs)
        except Exception as e:
            raise RuntimeError(
                f"Feature computation failed for candidate id={cand['id']!r}: {e}"
            ) from e
        rows.append(row)

    # Build DataFrame
    df = pd.DataFrame(rows)

    # Enforce column order: identity → baseline → artifact → ref → provenance
    fixed_cols = [
        "candidate_id", "text_raw",
        "target_tok_id", "target_family", "target_normalizer_id",
        "target_norm", "len_chars", "char_entropy",
        "zlib_bpc", "zlib_compression_ratio", "n_tokens_target",
        "tok_rank_mean", "tok_rank_min", "tok_rank_max",
        "merge_rank_mean", "merge_rank_max",
        "piece_score_mean", "piece_score_min", "piece_score_max",
    ]
    ref_cols = [c for c in df.columns if c.startswith("ref_")]
    tail_cols = ["extracted_at"]
    ordered = fixed_cols + sorted(ref_cols) + tail_cols
    # Any unexpected extra columns go at the end (defensive)
    extra = [c for c in df.columns if c not in ordered]
    df = df[ordered + extra]

    # Write Parquet
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{run_id}.parquet")
    df.to_parquet(out_path, index=False, engine="pyarrow")

    logger.info(
        "Wrote %d rows × %d columns → %s",
        len(df), len(df.columns), out_path,
    )
    logger.info("=== extract_features done ===")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.extract_features",
        description=(
            "Extract tokenizer-only features for candidate strings. "
            "Outputs one Parquet row per candidate."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config", metavar="YAML",
        help="Path to a project YAML config file. Provides defaults.",
    )
    p.add_argument(
        "--target-meta", metavar="FILE",
        help="Path to the TARGET tokenizer metadata.json.",
    )
    p.add_argument(
        "--ref-meta", metavar="FILE", nargs="*",
        help="Paths to one or more REFERENCE tokenizer metadata.json files.",
    )
    p.add_argument(
        "--candidates", metavar="FILE",
        help="Path to the JSONL candidate strings file.",
    )
    p.add_argument(
        "--run-id", metavar="ID",
        help="Run identifier used as the output Parquet filename stem.",
    )
    p.add_argument(
        "--output", metavar="DIR",
        help="Root features output directory.",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
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

    target_meta = candidates = output_dir = run_id = None
    ref_metas: Optional[List[str]] = None

    if args.config:
        from src.config import load_config  # type: ignore[import]
        cfg = load_config(args.config)
        target_meta  = os.path.join(cfg.paths.tokenizer_dir, "metadata.json")
        candidates   = cfg.corpus.canary.file
        output_dir   = cfg.paths.features_dir
        run_id       = cfg.run_id

    if args.target_meta:  target_meta  = args.target_meta
    if args.candidates:   candidates   = args.candidates
    if args.output:       output_dir   = args.output
    if args.run_id:       run_id       = args.run_id
    if args.ref_meta:     ref_metas    = args.ref_meta

    missing = [
        name for name, val in [
            ("--target-meta", target_meta),
            ("--candidates",  candidates),
            ("--output",      output_dir),
            ("--run-id",      run_id),
        ]
        if val is None
    ]
    if missing:
        parser.error(
            "Missing required values (provide via --config or explicit flags):\n  "
            + "\n  ".join(missing)
        )

    try:
        out_path = extract_features(
            target_metadata_path=target_meta,   # type: ignore[arg-type]
            candidates_path=candidates,          # type: ignore[arg-type]
            output_dir=output_dir,               # type: ignore[arg-type]
            run_id=run_id,                       # type: ignore[arg-type]
            ref_metadata_paths=ref_metas,
        )
    except (FileNotFoundError, ValueError, RuntimeError, ImportError) as e:
        logger.error("FATAL: %s", e)
        sys.exit(1)

    print(f"\nDone.  Features written to: {out_path}")


if __name__ == "__main__":
    main()
