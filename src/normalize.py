"""
normalize.py
------------
Tokenizer-based string normalization: s' = Normalize_Tok(s).

Extracts and applies the *tokenizer's own* normalizer so that all downstream
operations (tokenization, character-length measurement, BPC scoring) are
performed on the same normalized surface form.  Per SPEC.md §2.1, the same
s' must be used for tokenization, len_chars, and BPC.

Supported backends (dispatch order):
  1. ``tokenizers.Tokenizer``  (Rust-backed, raw)
  2. ``transformers.PreTrainedTokenizerFast``  (wraps a Rust tokenizer)
  3. ``transformers.PreTrainedTokenizer``  (slow) — **rejected loudly**
  4. ``sentencepiece.SentencePieceProcessor``

Public API (stable)
-------------------
  NormResult              Dataclass: .text, .normalizer_id, .changed
  get_normalizer_fn(tok)  Extract (callable, id_string) from a tokenizer
  normalize(text, tok)    Normalize one string
  normalize_batch(ts,tok) Normalize a list of strings
  describe_normalizer(tok)Human-readable summary
  selfcheck()             Smoke tests (``python -m src.normalize``)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, List, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NormResult:
    """Outcome of tokenizer-based normalization."""
    text: str           # normalized string s'
    normalizer_id: str  # which normalizer produced s'
    changed: bool       # s' != s


# ---------------------------------------------------------------------------
# Backend detection helpers
# ---------------------------------------------------------------------------

def _is_hf_rust_tokenizer(obj: Any) -> bool:
    """Raw ``tokenizers.Tokenizer`` from the Rust-backed library."""
    try:
        from tokenizers import Tokenizer  # type: ignore[import-untyped]
        return isinstance(obj, Tokenizer)
    except ImportError:
        return False


def _is_hf_fast_pretrained(obj: Any) -> bool:
    """``transformers.PreTrainedTokenizerFast`` (wraps a Rust backend)."""
    try:
        from transformers import PreTrainedTokenizerFast  # type: ignore[import-untyped]
        return isinstance(obj, PreTrainedTokenizerFast)
    except ImportError:
        return False


def _is_hf_slow_pretrained(obj: Any) -> bool:
    """``transformers.PreTrainedTokenizer`` (pure-Python, no Rust backend).

    Must NOT match Fast tokenizers.  In ``transformers``, Fast and Slow do not
    share an inheritance chain (both extend ``PreTrainedTokenizerBase``
    independently), so a simple isinstance check is safe.
    """
    try:
        from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast  # type: ignore[import-untyped]
        return isinstance(obj, PreTrainedTokenizer) and not isinstance(obj, PreTrainedTokenizerFast)
    except ImportError:
        return False


def _is_sp_processor(obj: Any) -> bool:
    try:
        import sentencepiece as spm  # type: ignore[import-untyped]
        return isinstance(obj, spm.SentencePieceProcessor)
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Rust tokenizers.Tokenizer path (also used by PreTrainedTokenizerFast)
# ---------------------------------------------------------------------------

def _rust_normalizer_id(rust_tokenizer: Any) -> str:
    """Human-readable id from a Rust ``tokenizers.Tokenizer``'s normalizer."""
    norm = rust_tokenizer.normalizer
    if norm is None:
        return "none"
    return repr(norm)


def _get_rust_normalizer_fn(rust_tokenizer: Any) -> Tuple[Callable[[str], str], str]:
    """Extract normalizer from a Rust ``tokenizers.Tokenizer``."""
    norm = rust_tokenizer.normalizer
    nid = _rust_normalizer_id(rust_tokenizer)

    if norm is None:
        logger.info("Rust tokenizer has normalizer=None — identity pass-through")
        return _identity, nid

    def _normalize(s: str) -> str:
        if not s:
            return s
        return norm.normalize_str(s)

    return _normalize, nid


# ---------------------------------------------------------------------------
# transformers.PreTrainedTokenizerFast path
# ---------------------------------------------------------------------------

def _get_fast_pretrained_normalizer_fn(tokenizer: Any) -> Tuple[Callable[[str], str], str]:
    """Extract normalizer from ``PreTrainedTokenizerFast`` via its Rust backend.

    The Fast tokenizer stores a ``tokenizers.Tokenizer`` in
    ``.backend_tokenizer``.  We delegate to the Rust-level normalizer
    so the normalization we apply is *exactly* what happens inside
    ``tokenizer.encode()``.

    Raises ``RuntimeError`` if the backend is missing (should never happen
    for a properly constructed fast tokenizer, but we refuse to guess).
    """
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        raise RuntimeError(
            f"PreTrainedTokenizerFast ({type(tokenizer).__name__}) has no "
            f"backend_tokenizer attribute.  Cannot extract normalizer.  "
            f"This should not happen — the tokenizer may be corrupted."
        )

    fn, rust_nid = _get_rust_normalizer_fn(backend)

    # Tag the id so logs show provenance
    tok_class = type(tokenizer).__name__
    nid = f"{tok_class}->{rust_nid}"
    return fn, nid


# ---------------------------------------------------------------------------
# transformers.PreTrainedTokenizer (slow) — always rejected
# ---------------------------------------------------------------------------

def _reject_slow_pretrained(tokenizer: Any) -> None:
    """Fail loudly for slow tokenizers.

    Slow tokenizers do not expose a programmatically accessible normalizer.
    Some (e.g. LlamaTokenizer) wrap SentencePiece internally, but extracting
    normalization from that internal object is fragile and not guaranteed to
    match what ``tokenizer.encode()`` actually does.  Per SPEC.md §2.1,
    we must not invent normalization behavior.
    """
    raise TypeError(
        f"Slow tokenizer ({type(tokenizer).__name__}) is not supported.  "
        f"Slow (pure-Python) tokenizers do not expose a Rust-level normalizer, "
        f"so we cannot guarantee Normalize_Tok(s) matches what encode() does.  "
        f"Use the fast version instead:\n"
        f"  from transformers import AutoTokenizer\n"
        f"  tok = AutoTokenizer.from_pretrained('<model>', use_fast=True)\n"
        f"If no fast variant exists for this model, use the raw "
        f"tokenizers.Tokenizer or sentencepiece.SentencePieceProcessor directly."
    )


# ---------------------------------------------------------------------------
# SentencePiece path
# ---------------------------------------------------------------------------

def _get_sp_normalizer_fn(sp: Any) -> Tuple[Callable[[str], str], str]:
    normalize_method = getattr(sp, "Normalize", None) or getattr(sp, "normalize", None)

    if normalize_method is None:
        raise AttributeError(
            "This SentencePieceProcessor build does not expose Normalize().  "
            "Cannot extract the tokenizer's normalizer.  "
            "Upgrade sentencepiece (pip install -U sentencepiece) or use "
            "the HuggingFace tokenizers backend."
        )

    nid = _sp_normalizer_id(sp)

    def _normalize(s: str) -> str:
        if not s:
            return s
        return normalize_method(s)

    return _normalize, nid


def _sp_normalizer_id(sp: Any) -> str:
    """Best-effort extraction of the SP normalization rule name."""
    try:
        import sentencepiece.sentencepiece_model_pb2 as sp_pb2  # type: ignore[import-untyped]
        mp = sp_pb2.ModelProto()
        mp.ParseFromString(sp.serialized_model_proto())
        rule = mp.normalizer_spec.name
        if rule:
            return f"sentencepiece({rule})"
    except Exception:
        pass
    return "sentencepiece(unknown_rule)"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _identity(s: str) -> str:
    return s


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_normalizer_fn(tokenizer: Any) -> Tuple[Callable[[str], str], str]:
    """
    Extract the normalization function from a tokenizer object.

    Parameters
    ----------
    tokenizer
        One of:
        - ``tokenizers.Tokenizer``  (Rust-backed)
        - ``transformers.PreTrainedTokenizerFast``
        - ``sentencepiece.SentencePieceProcessor``

        Passing a slow ``transformers.PreTrainedTokenizer`` raises TypeError
        with guidance to switch to the fast variant.

    Returns
    -------
    (normalize_fn, normalizer_id)
        normalize_fn : str -> str
        normalizer_id : human-readable string

    Raises
    ------
    TypeError
        For slow tokenizers or unrecognized types.
    RuntimeError
        If a fast tokenizer's backend is unexpectedly missing.
    AttributeError
        If SentencePiece doesn't expose Normalize().
    """
    # 1) Raw Rust tokenizer (most specific — check before transformers wrappers)
    if _is_hf_rust_tokenizer(tokenizer):
        return _get_rust_normalizer_fn(tokenizer)

    # 2) transformers Fast tokenizer (wraps a Rust backend)
    if _is_hf_fast_pretrained(tokenizer):
        return _get_fast_pretrained_normalizer_fn(tokenizer)

    # 3) transformers Slow tokenizer — reject
    if _is_hf_slow_pretrained(tokenizer):
        _reject_slow_pretrained(tokenizer)

    # 4) SentencePiece
    if _is_sp_processor(tokenizer):
        return _get_sp_normalizer_fn(tokenizer)

    raise TypeError(
        f"Unsupported tokenizer type: {type(tokenizer).__qualname__}.  "
        f"Expected one of: tokenizers.Tokenizer, "
        f"transformers.PreTrainedTokenizerFast, "
        f"sentencepiece.SentencePieceProcessor."
    )


def normalize(text: str, tokenizer: Any) -> NormResult:
    """
    Apply tokenizer-based normalization to a single string.

    Parameters
    ----------
    text : str
        Raw input string s.
    tokenizer
        A trained tokenizer (any supported type).

    Returns
    -------
    NormResult
        .text is the normalized string s',
        .normalizer_id names the normalizer,
        .changed is True if s' != s.
    """
    fn, nid = get_normalizer_fn(tokenizer)
    normed = fn(text)
    return NormResult(text=normed, normalizer_id=nid, changed=(normed != text))


def normalize_batch(texts: List[str], tokenizer: Any) -> List[NormResult]:
    """
    Apply tokenizer-based normalization to a list of strings.

    Extracts the normalizer once and reuses it across the batch.
    """
    fn, nid = get_normalizer_fn(tokenizer)
    results = []
    for t in texts:
        normed = fn(t)
        results.append(NormResult(text=normed, normalizer_id=nid, changed=(normed != t)))
    return results


def describe_normalizer(tokenizer: Any) -> str:
    """Return a human-readable one-line description of the tokenizer's normalizer."""
    _, nid = get_normalizer_fn(tokenizer)
    return nid


# ---------------------------------------------------------------------------
# Self-checks
# ---------------------------------------------------------------------------

def selfcheck() -> None:
    """
    Concrete smoke tests for the three dispatch paths we actually use.

    Check 1: GPT-2 via transformers (PreTrainedTokenizerFast, normalizer=none)
    Check 2: BERT-base-uncased via transformers (PreTrainedTokenizerFast, BertNormalizer)
    Check 3: SentencePieceProcessor (or loud failure with install guidance)

    Also validates:
    - Unsupported type → TypeError
    - Slow tokenizer → TypeError with guidance
    - Empty string handling
    - Batch API
    """
    import sys

    passed, failed, skipped = 0, 0, 0

    def _assert(cond: bool, label: str) -> None:
        nonlocal passed, failed
        if cond:
            passed += 1
            print(f"  PASS  {label}")
        else:
            failed += 1
            print(f"  FAIL  {label}")

    def _skip(label: str, reason: str) -> None:
        nonlocal skipped
        skipped += 1
        print(f"  SKIP  {label} ({reason})")

    print("normalize.py self-checks")
    print("=" * 60)

    # ---- Generic: unsupported type raises TypeError ----
    try:
        get_normalizer_fn("not_a_tokenizer")
        _assert(False, "TypeError for unsupported type")
    except TypeError:
        _assert(True, "TypeError for unsupported type")

    # ---- Generic: slow tokenizer raises TypeError with guidance ----
    try:
        from transformers import PreTrainedTokenizer  # type: ignore[import-untyped]

        class _DummySlow(PreTrainedTokenizer):
            """Minimal slow tokenizer stub for testing rejection."""
            @property
            def vocab_size(self) -> int:
                return 0
            def _tokenize(self, text: str) -> list:
                return []
            def _convert_token_to_id(self, token: str) -> int:
                return 0
            def _convert_id_to_token(self, index: int) -> str:
                return ""
            def get_vocab(self) -> dict:
                return {}

        try:
            get_normalizer_fn(_DummySlow())
            _assert(False, "TypeError for slow tokenizer")
        except TypeError as e:
            _assert("use_fast=True" in str(e), "TypeError for slow tokenizer (with guidance)")

    except ImportError:
        _skip("Slow tokenizer rejection", "transformers not installed")

    # ---- Check 1: GPT-2 (PreTrainedTokenizerFast, normalizer=none) ----
    try:
        from transformers import AutoTokenizer  # type: ignore[import-untyped]

        gpt2 = AutoTokenizer.from_pretrained("openai-community/gpt2", use_fast=True)
        fn, nid = get_normalizer_fn(gpt2)
        _assert("none" in nid, "GPT-2 normalizer_id contains 'none'")

        r = normalize("Hello, world!", gpt2)
        _assert(r.text == "Hello, world!", "GPT-2 identity on ASCII")
        _assert(r.changed is False, "GPT-2 changed=False on ASCII")

        r_empty = normalize("", gpt2)
        _assert(r_empty.text == "", "GPT-2 empty string -> empty string")
        _assert(r_empty.changed is False, "GPT-2 empty changed=False")

        desc = describe_normalizer(gpt2)
        _assert(isinstance(desc, str) and "none" in desc, "GPT-2 describe_normalizer")

    except ImportError:
        _skip("GPT-2 checks", "transformers not installed")
    except OSError as e:
        _skip("GPT-2 checks", f"model download failed: {e}")

    # ---- Check 2: BERT-base-uncased (BertNormalizer: lowercases + strips accents) ----
    try:
        from transformers import AutoTokenizer  # type: ignore[import-untyped]

        bert = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased", use_fast=True)
        fn, nid = get_normalizer_fn(bert)
        _assert("none" not in nid.lower() or "Bert" in nid, "BERT has a real normalizer")

        # BertNormalizer lowercases
        r_case = normalize("HELLO", bert)
        _assert(r_case.text == "hello", "BERT lowercases 'HELLO' → 'hello'")
        _assert(r_case.changed is True, "BERT changed=True for uppercase")

        # BertNormalizer strips accents (default for uncased)
        r_accent = normalize("café", bert)
        _assert("é" not in r_accent.text, "BERT strips accent from 'café'")
        _assert(r_accent.changed is True, "BERT changed=True for accented input")

        # Batch
        batch = normalize_batch(["HELLO", "café", "plain"], bert)
        _assert(len(batch) == 3, "BERT batch returns 3 results")
        _assert(batch[2].changed is False, "BERT 'plain' unchanged")

    except ImportError:
        _skip("BERT checks", "transformers not installed")
    except OSError as e:
        _skip("BERT checks", f"model download failed: {e}")

    # ---- Check 3: SentencePiece ----
    try:
        import sentencepiece as spm  # type: ignore[import-untyped]
        sp = spm.SentencePieceProcessor()
        _assert(_is_sp_processor(sp), "SP processor detection")

        # Without a trained model we can't call Normalize, but we can verify
        # the method exists on the class (version check).
        has_normalize = hasattr(sp, "Normalize") or hasattr(sp, "normalize")
        if has_normalize:
            _assert(True, "SP exposes Normalize method")
        else:
            print("  WARN  SP installed but Normalize() not exposed — "
                  "upgrade: pip install -U sentencepiece")

    except ImportError:
        print("  INFO  sentencepiece not installed.  If needed:")
        print("        pip install sentencepiece")
        _skip("SP checks", "sentencepiece not installed")

    # ---- Check: raw tokenizers.Tokenizer still works ----
    try:
        from tokenizers import Tokenizer, normalizers, models  # type: ignore[import-untyped]

        tok_bare = Tokenizer(models.BPE())
        fn, nid = get_normalizer_fn(tok_bare)
        _assert(nid == "none", "Raw Rust BPE normalizer_id='none'")
        _assert(fn("test") == "test", "Raw Rust BPE identity")

        tok_nfkc = Tokenizer(models.BPE())
        tok_nfkc.normalizer = normalizers.NFKC()  # type: ignore[assignment]
        r = normalize("\uff21", tok_nfkc)   # fullwidth A → A
        _assert(r.text == "A", "Raw Rust NFKC normalizes fullwidth A")
        _assert(r.changed is True, "Raw Rust NFKC changed=True")

    except ImportError:
        _skip("Raw Rust tokenizer checks", "tokenizers not installed")

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    selfcheck()
