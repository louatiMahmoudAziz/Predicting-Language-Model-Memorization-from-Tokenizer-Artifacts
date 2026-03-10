"""
train_tokenizer.py
------------------
Train a BPE or SentencePiece Unigram tokenizer on a corpus file and save
all artifacts under tokenizers/<tok_id>/.

Supported algorithms
--------------------
  bpe      — Hugging Face ``tokenizers`` (Rust-backed BPE).
             Artifacts: tokenizer.json, vocab.json, merges.txt, metadata.json
  unigram  — Google SentencePiece Unigram.
             Artifacts: <tok_id>.model, <tok_id>.vocab, metadata.json

Design invariants
-----------------
- Normalization is NEVER invented.  Whatever the library applies is recorded
  verbatim in metadata.json.  Downstream modules read normalization from
  metadata; they never guess.
- Artifacts are self-contained: metadata.json carries enough information to
  load the tokenizer without re-reading the config.
- Merge-rank and piece-score data are preserved in stable files so
  extract_features.py can compute rank/score statistics without re-training.
- Training is deterministic given the same corpus and config (both libraries
  produce deterministic output given fixed hyperparameters and a single-threaded
  training pass; SentencePiece requires --num_threads=1 for this).

CLI
---
  python -m src.train_tokenizer --config configs/colab_mini.yaml [overrides]
  python -m src.train_tokenizer --help
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import os
import shutil
import sys
from typing import Dict, List, Optional

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
# Corpus iterator (shared by both trainers)
# ---------------------------------------------------------------------------

def _iter_lines(corpus_path: str):
    """Yield non-empty lines from a corpus file."""
    with open(corpus_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line:
                yield line


def _count_lines(corpus_path: str) -> int:
    return sum(1 for _ in _iter_lines(corpus_path))


# ---------------------------------------------------------------------------
# BPE trainer (Hugging Face tokenizers)
# ---------------------------------------------------------------------------

def _train_bpe(
    corpus_path: str,
    output_dir: str,
    tok_id: str,
    vocab_size: int,
    min_frequency: int,
    special_tokens: List[str],
) -> Dict:
    """
    Train a BPE tokenizer using the Hugging Face ``tokenizers`` library.

    Normalization: NONE by default (matches GPT-2 / LLaMA convention).
    Pre-tokenization: ByteLevel (byte-fallback, no unknown tokens).
    Decoder: ByteLevel.

    The ByteLevel pre-tokenizer handles all Unicode via byte encoding;
    there is no separate Unicode normalizer, so normalizer_id = "none".

    Artifacts saved:
      tokenizer.json   — full Rust tokenizer (load with tokenizers.Tokenizer)
      vocab.json       — {piece: id} mapping
      merges.txt       — BPE merge rules, one per line, rank-ordered (0-indexed)

    Returns metadata dict (not yet saved; caller writes metadata.json).
    """
    try:
        from tokenizers import Tokenizer  # type: ignore[import-untyped]
        from tokenizers.models import BPE  # type: ignore[import-untyped]
        from tokenizers.trainers import BpeTrainer  # type: ignore[import-untyped]
        from tokenizers.pre_tokenizers import ByteLevel  # type: ignore[import-untyped]
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            f"Hugging Face 'tokenizers' library is required for BPE training: {e}\n"
            f"Install with: pip install tokenizers"
        ) from e

    logger.info("Training BPE tokenizer (vocab_size=%d, min_freq=%d)", vocab_size, min_frequency)

    tok = Tokenizer(BPE(unk_token="<unk>" if "<unk>" in special_tokens else None))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=False,
    )

    tok.train_from_iterator(_iter_lines(corpus_path), trainer=trainer)

    actual_vocab_size = tok.get_vocab_size()
    logger.info("BPE training complete. Actual vocab size: %d", actual_vocab_size)

    # --- Save artifacts ---

    # tokenizer.json (full Rust tokenizer)
    tok_json_path = os.path.join(output_dir, "tokenizer.json")
    tok.save(tok_json_path)
    logger.info("Saved tokenizer.json → %s", tok_json_path)

    # vocab.json  {piece: id}
    vocab = tok.get_vocab(with_added_tokens=True)
    vocab_path = os.path.join(output_dir, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as fh:
        json.dump(vocab, fh, indent=2, ensure_ascii=False, sort_keys=False)
    logger.info("Saved vocab.json  → %s  (%d entries)", vocab_path, len(vocab))

    # merges.txt — rank-ordered merge rules via model.save()
    # The BPE model does not expose a .merges Python attribute; the canonical
    # way to export merge rules is model.save(folder, prefix), which writes
    # two files: <prefix>-vocab.json and <prefix>-merges.txt.
    # We use prefix=None so the files are named "vocab.json" and "merges.txt".
    merges_path = os.path.join(output_dir, "merges.txt")
    try:
        saved_files = tok.model.save(output_dir, prefix=None)
        # saved_files is a list of written paths; merges.txt is one of them
        n_merges = 0
        if os.path.isfile(merges_path):
            with open(merges_path, encoding="utf-8") as fh:
                n_merges = sum(
                    1 for ln in fh if ln.strip() and not ln.startswith("#")
                )
            logger.info("Saved merges.txt  → %s  (%d merges)", merges_path, n_merges)
        else:
            # model.save may write it under a different name; log what was written
            logger.warning(
                "model.save() wrote %s but merges.txt not found at expected path.",
                saved_files,
            )
            merges_path = None
            n_merges = 0
    except Exception as e:
        logger.warning("Could not save merges via model.save(): %s", e)
        merges_path = None
        n_merges = 0

    # Normalization: BPE via ByteLevel pre-tokenizer has no Unicode normalizer.
    normalizer_id = "none"
    normalizer_detail = (
        "ByteLevel pre-tokenizer encodes all Unicode as UTF-8 bytes. "
        "No Unicode normalizer is applied."
    )

    return {
        "family": "bpe",
        "algorithm": "BPE",
        "library": "huggingface/tokenizers",
        "vocab_size_requested": vocab_size,
        "vocab_size_actual": actual_vocab_size,
        "min_frequency": min_frequency,
        "special_tokens": special_tokens,
        "normalizer_id": normalizer_id,
        "normalizer_detail": normalizer_detail,
        "pre_tokenizer": "ByteLevel(add_prefix_space=False)",
        "artifacts": {
            "tokenizer_json": os.path.abspath(tok_json_path),
            "vocab_json": os.path.abspath(vocab_path),
            "merges_txt": os.path.abspath(merges_path) if merges_path else None,
            "n_merges": n_merges,
        },
    }


# ---------------------------------------------------------------------------
# SentencePiece Unigram trainer
# ---------------------------------------------------------------------------

def _train_unigram(
    corpus_path: str,
    output_dir: str,
    tok_id: str,
    vocab_size: int,
    min_frequency: int,
    special_tokens: List[str],
) -> Dict:
    """
    Train a SentencePiece Unigram tokenizer.

    Normalization: NFKC (SentencePiece default).
    The normalization rule name is read back from the trained model proto
    and recorded verbatim in metadata.

    num_threads=1 is set explicitly for reproducibility.

    Artifacts saved:
      <tok_id>.model   — binary SentencePiece model
      <tok_id>.vocab   — tab-separated (piece, score) in vocab-id order

    Returns metadata dict (not yet saved; caller writes metadata.json).
    """
    try:
        import sentencepiece as spm  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            f"'sentencepiece' library is required for unigram training: {e}\n"
            f"Install with: pip install sentencepiece"
        ) from e

    model_prefix = os.path.join(output_dir, tok_id)
    model_path = model_prefix + ".model"
    vocab_path = model_prefix + ".vocab"

    logger.info(
        "Training SentencePiece Unigram (vocab_size=%d, min_freq=%d)",
        vocab_size, min_frequency,
    )

    # Build control symbols list (everything except <unk>, which SP handles natively)
    control_symbols = [t for t in special_tokens if t != "<unk>"]
    # SP uses unk_piece separately; if caller included <unk>, it's handled via unk_piece
    unk_piece = "<unk>" if "<unk>" in special_tokens else None

    train_kwargs = dict(
        input=corpus_path,
        model_prefix=model_prefix,
        model_type="unigram",
        vocab_size=vocab_size,
        character_coverage=0.9995,
        pad_id=special_tokens.index("<pad>") if "<pad>" in special_tokens else -1,
        bos_id=special_tokens.index("<s>") if "<s>" in special_tokens else -1,
        eos_id=special_tokens.index("</s>") if "</s>" in special_tokens else -1,
        unk_id=special_tokens.index("<unk>") if "<unk>" in special_tokens else 0,
        pad_piece="<pad>" if "<pad>" in special_tokens else "",
        bos_piece="<s>" if "<s>" in special_tokens else "",
        eos_piece="</s>" if "</s>" in special_tokens else "",
        unk_piece=unk_piece or "<unk>",
        control_symbols=",".join(control_symbols) if control_symbols else "",
        # Reproducibility: single-threaded EM
        num_threads=1,
        # Normalization: NFKC (SP default, explicitly set)
        normalization_rule_name="nfkc",
        # Note: SP unigram does not have a direct token-frequency threshold
        # equivalent to BPE's min_frequency.  Frequency is implicitly handled
        # by the EM algorithm; low-frequency pieces are pruned.  min_frequency
        # is recorded in metadata for documentation only, not passed to SP.
        # Allow SP to produce a smaller vocab than requested when the corpus
        # does not have enough unique character sequences.
        hard_vocab_limit=False,
    )

    # Filter empty string kwargs (SP trainer rejects empty string for some fields)
    train_kwargs = {k: v for k, v in train_kwargs.items() if v != ""}

    spm.SentencePieceTrainer.train(**train_kwargs)

    if not os.path.isfile(model_path):
        raise RuntimeError(
            f"SentencePiece training completed but model file not found: {model_path}"
        )

    logger.info("SP training complete. Model saved: %s", model_path)

    # Read back actual normalization rule from trained model proto
    normalizer_id = _sp_read_normalizer(model_path)
    actual_vocab_size = _sp_read_vocab_size(model_path)
    logger.info(
        "SP actual vocab_size=%d  normalizer=%s", actual_vocab_size, normalizer_id
    )

    # Export vocab file as tab-separated (piece \t score) if not already written
    if not os.path.isfile(vocab_path):
        _sp_export_vocab(model_path, vocab_path)
    logger.info("Saved vocab file → %s", vocab_path)

    return {
        "family": "unigram",
        "algorithm": "SentencePiece_Unigram",
        "library": "google/sentencepiece",
        "vocab_size_requested": vocab_size,
        "vocab_size_actual": actual_vocab_size,
        "min_frequency": min_frequency,
        "special_tokens": special_tokens,
        "normalizer_id": normalizer_id,
        "normalizer_detail": (
            f"SentencePiece built-in normalization rule: {normalizer_id!r}. "
            f"Applied inside sp.encode() and exposed via sp.Normalize()."
        ),
        "artifacts": {
            "model": os.path.abspath(model_path),
            "vocab": os.path.abspath(vocab_path),
        },
    }


def _sp_read_normalizer(model_path: str) -> str:
    """Read the normalization rule name from a trained SP model proto."""
    try:
        import sentencepiece.sentencepiece_model_pb2 as sp_pb2  # type: ignore[import-untyped]
        mp = sp_pb2.ModelProto()
        with open(model_path, "rb") as fh:
            mp.ParseFromString(fh.read())
        rule = mp.normalizer_spec.name
        return rule if rule else "unknown"
    except Exception as e:
        logger.warning("Could not read normalizer from SP proto: %s", e)
        return "unknown"


def _sp_read_vocab_size(model_path: str) -> int:
    """Read actual vocab size from a trained SP model."""
    try:
        import sentencepiece as spm  # type: ignore[import-untyped]
        sp = spm.SentencePieceProcessor(model_file=model_path)
        return sp.get_piece_size()
    except Exception as e:
        logger.warning("Could not read vocab size from SP model: %s", e)
        return -1


def _sp_export_vocab(model_path: str, vocab_path: str) -> None:
    """Export SP vocabulary as tab-separated (piece, score) in vocab-id order."""
    try:
        import sentencepiece as spm  # type: ignore[import-untyped]
        sp = spm.SentencePieceProcessor(model_file=model_path)
        n = sp.get_piece_size()
        with open(vocab_path, "w", encoding="utf-8") as fh:
            for i in range(n):
                piece = sp.id_to_piece(i)
                score = sp.get_score(i)
                fh.write(f"{piece}\t{score}\n")
    except Exception as e:
        logger.warning("Could not export SP vocab: %s", e)


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train_tokenizer(
    corpus_path: str,
    tok_type: str,
    vocab_size: int,
    min_frequency: int,
    special_tokens: List[str],
    output_dir: str,
    tok_id: str,
    config_snapshot: Optional[Dict] = None,
) -> str:
    """
    Train a tokenizer and write all artifacts to output_dir/tok_id/.

    Parameters
    ----------
    corpus_path    : path to the training corpus text file (one line per sample)
    tok_type       : "bpe" or "unigram"
    vocab_size     : target vocabulary size
    min_frequency  : minimum token frequency (BPE) / minimum sentence count (SP)
    special_tokens : ordered list of special tokens to add
    output_dir     : root tokenizer directory (tok_id sub-dir appended)
    tok_id         : unique identifier for this tokenizer artifact set
    config_snapshot: optional dict written verbatim into metadata (for reproducibility)

    Returns
    -------
    str  Path to the written metadata.json file.
    """
    if not os.path.isfile(corpus_path):
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    tok_type = tok_type.lower()
    if tok_type not in ("bpe", "unigram"):
        raise ValueError(
            f"tok_type must be 'bpe' or 'unigram', got {tok_type!r}."
        )

    if vocab_size < 256:
        raise ValueError(
            f"vocab_size={vocab_size} is unusually small (< 256). "
            f"Pass --vocab-size explicitly if this is intentional."
        )

    if not special_tokens:
        logger.warning(
            "No special tokens provided. This is unusual; "
            "most models need at least <unk>."
        )

    # Validate no duplicate special tokens
    seen = set()
    for t in special_tokens:
        if t in seen:
            raise ValueError(f"Duplicate special token: {t!r}")
        seen.add(t)

    tok_dir = os.path.join(output_dir, tok_id)
    os.makedirs(tok_dir, exist_ok=True)
    logger.info("Tokenizer output dir: %s", tok_dir)

    corpus_sha = _sha256(corpus_path)
    n_lines = _count_lines(corpus_path)
    logger.info("Corpus: %s  (%d lines, sha256=%s...)", corpus_path, n_lines, corpus_sha[:16])

    # Dispatch to family-specific trainer
    if tok_type == "bpe":
        family_meta = _train_bpe(
            corpus_path=corpus_path,
            output_dir=tok_dir,
            tok_id=tok_id,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
        )
    else:
        family_meta = _train_unigram(
            corpus_path=corpus_path,
            output_dir=tok_dir,
            tok_id=tok_id,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
        )

    # Build and write metadata.json
    metadata = {
        "tok_id": tok_id,
        "tok_type": tok_type,
        "family": family_meta["family"],
        "algorithm": family_meta["algorithm"],
        "library": family_meta["library"],
        "vocab_size_requested": family_meta["vocab_size_requested"],
        "vocab_size_actual": family_meta["vocab_size_actual"],
        "min_frequency": family_meta["min_frequency"],
        "special_tokens": family_meta["special_tokens"],
        "normalizer_id": family_meta["normalizer_id"],
        "normalizer_detail": family_meta["normalizer_detail"],
        # BPE only
        "pre_tokenizer": family_meta.get("pre_tokenizer"),
        # Corpus provenance
        "corpus_path": os.path.abspath(corpus_path),
        "corpus_sha256": corpus_sha,
        "corpus_n_lines": n_lines,
        # Timestamps
        "trained_at": datetime.datetime.utcnow().isoformat() + "Z",
        # Artifacts (family-specific paths)
        "artifacts": family_meta["artifacts"],
        # Config snapshot for exact reproducibility
        "config_snapshot": config_snapshot,
    }

    metadata_path = os.path.join(tok_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)

    logger.info("Wrote metadata.json → %s", metadata_path)
    logger.info("=== train_tokenizer done: %s ===", tok_id)

    return metadata_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.train_tokenizer",
        description=(
            "Train a BPE or SentencePiece Unigram tokenizer and save artifacts. "
            "All arguments can be supplied via --config; CLI flags override config values."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config", metavar="YAML",
        help="Path to a project YAML config file. Provides defaults for all fields.",
    )
    p.add_argument(
        "--corpus", metavar="FILE",
        help="Path to the training corpus text file (e.g. D_clean.txt).",
    )
    p.add_argument(
        "--type", dest="tok_type", choices=["bpe", "unigram"],
        help="Tokenizer algorithm.",
    )
    p.add_argument(
        "--vocab-size", type=int, metavar="N",
        help="Target vocabulary size.",
    )
    p.add_argument(
        "--min-frequency", type=int, metavar="N", default=None,
        help="Minimum token frequency (BPE) or minimum sentence count (unigram).",
    )
    p.add_argument(
        "--special-tokens", nargs="*", metavar="TOK",
        help="Ordered list of special tokens, e.g. --special-tokens '<unk>' '<pad>'.",
    )
    p.add_argument(
        "--tok-id", metavar="ID",
        help="Unique identifier for this tokenizer artifact set (output sub-directory name).",
    )
    p.add_argument(
        "--output", metavar="DIR",
        help="Root tokenizer directory (tok_id sub-dir appended). Overrides config paths.tokenizer_dir.",
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

    # --- Resolve values: config first, then CLI overrides ---
    corpus_path = tok_type = output_dir = tok_id = None
    vocab_size = min_frequency = None
    special_tokens: Optional[List[str]] = None
    config_snapshot: Optional[Dict] = None

    if args.config:
        from src.config import load_config  # type: ignore[import]
        cfg = load_config(args.config)
        # Use D_clean.txt from the corpus_dir as the default training corpus
        corpus_path   = os.path.join(
            cfg.paths.corpus_dir, cfg.run_id, "D_clean.txt"
        )
        tok_type      = cfg.tokenizer.type
        vocab_size    = cfg.tokenizer.vocab_size
        min_frequency = cfg.tokenizer.min_frequency
        special_tokens = cfg.tokenizer.special_tokens
        output_dir    = os.path.dirname(cfg.paths.tokenizer_dir)
        tok_id        = os.path.basename(cfg.paths.tokenizer_dir)

        # Snapshot the tokenizer section of the config for metadata
        config_snapshot = {
            "run_id":       cfg.run_id,
            "seed":         cfg.seed,
            "tok_type":     cfg.tokenizer.type,
            "vocab_size":   cfg.tokenizer.vocab_size,
            "min_frequency":cfg.tokenizer.min_frequency,
            "special_tokens":cfg.tokenizer.special_tokens,
            "config_file":  os.path.abspath(args.config),
        }

    # CLI overrides
    if args.corpus:        corpus_path   = args.corpus
    if args.tok_type:      tok_type      = args.tok_type
    if args.vocab_size:    vocab_size    = args.vocab_size
    if args.min_frequency is not None:
                           min_frequency = args.min_frequency
    if args.special_tokens is not None:
                           special_tokens = args.special_tokens
    if args.output:        output_dir    = args.output
    if args.tok_id:        tok_id        = args.tok_id

    # Validate all required fields
    missing = [
        name for name, val in [
            ("--corpus",          corpus_path),
            ("--type",            tok_type),
            ("--vocab-size",      vocab_size),
            ("--tok-id",          tok_id),
            ("--output",          output_dir),
        ]
        if val is None
    ]
    if missing:
        parser.error(
            "Missing required values (provide via --config or explicit flags):\n  "
            + "\n  ".join(missing)
        )

    if min_frequency is None:
        min_frequency = 2
        logger.info("min_frequency not set; defaulting to 2")

    if special_tokens is None:
        special_tokens = ["<unk>", "<pad>", "<s>", "</s>"]
        logger.info("special_tokens not set; defaulting to %s", special_tokens)

    try:
        metadata_path = train_tokenizer(
            corpus_path=corpus_path,    # type: ignore[arg-type]
            tok_type=tok_type,          # type: ignore[arg-type]
            vocab_size=vocab_size,      # type: ignore[arg-type]
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            output_dir=output_dir,      # type: ignore[arg-type]
            tok_id=tok_id,              # type: ignore[arg-type]
            config_snapshot=config_snapshot,
        )
    except (FileNotFoundError, ValueError, RuntimeError, ImportError) as e:
        logger.error("FATAL: %s", e)
        sys.exit(1)

    print(f"\nDone.  metadata.json written to: {metadata_path}")


if __name__ == "__main__":
    main()
