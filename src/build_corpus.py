"""
build_corpus.py
---------------
Build D_clean and D_canary training corpora from a raw text source.

Definitions (SPEC.md §3, §4):
  D_clean   — base corpus trimmed to budget, canary-free.
  D_canary  — D_clean + injected canaries at a repetition schedule.

Outputs (under <corpus_dir>/<run_id>/):
  D_clean.txt    — clean corpus lines, one per line
  D_canary.txt   — canary-injected corpus lines, one per line
  canaries.json  — canary manifest (one record per (id, repetition_level) pair)
  manifest.json  — full run metadata for reproducibility

Injection design
----------------
Positions are sampled ENTIRELY from the clean corpus index space [0, n_clean]
before any insertion is applied.  The sampling space is NEVER updated after
each insertion.  Specifically:

  - Each insertion point p is an integer in range(0, n_clean + 1), meaning
    "insert this canary line immediately before clean line p"
    (p == n_clean means append after the last clean line).
  - All insertion points for all canaries at all repetition levels are
    pre-computed and sorted before the output is assembled.
  - The final output is built in a single deterministic merge pass.

Repetition semantics
--------------------
If the schedule is [1, 10, 100] and canary "c000" has no per-entry override,
three separate (id, repetition_level) pairs are created:
  ("c000", 1)   → injected 1   time
  ("c000", 10)  → injected 10  times
  ("c000", 100) → injected 100 times
canaries.json records each pair as a separate entry with an explicit
repetition_level field.

CLI:
  python -m src.build_corpus --config configs/colab_mini.yaml [overrides]
  python -m src.build_corpus --help
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Canary:
    id: str           # stable identifier (from input file or auto-assigned)
    text: str         # raw canary string (NOT yet injected into template)
    repetitions: int  # how many times this canary is injected into D_canary


@dataclass
class CanaryRecord:
    """
    One (id, repetition_level) pair and its injection positions in D_canary.

    repetition_level is the explicit number of times this canary was injected
    (e.g. 1, 10, or 100 for a schedule of [1, 10, 100]).
    len(positions) == repetition_level always.
    positions are 0-based line indices in D_canary.txt.
    """
    id: str
    text: str
    repetition_level: int       # explicit: how many times this entry is injected
    injected_text: str          # line as written into D_canary (after template)
    positions: List[int]        # line indices in D_canary where it appears


@dataclass
class Manifest:
    run_id: str
    seed: int
    budget_type: str
    budget_value: int
    base_source: str
    base_source_sha256: str
    canary_file: str
    canary_file_sha256: str
    repetitions: List[int]
    template: str
    n_clean_lines: int
    n_clean_chars: int
    n_canary_lines: int
    n_canary_chars: int
    n_canaries: int
    output_dir: str


# ---------------------------------------------------------------------------
# Canary file loading + validation
# ---------------------------------------------------------------------------

def _load_canaries(path: str, repetitions: List[int]) -> List[Canary]:
    """
    Load canaries from a JSONL file.

    Each line must be a JSON object with at least a "text" field.
    Optional "id" field; if absent, auto-assigned as "canary_{i:05d}".
    Optional "repetitions" field; overrides the schedule-level default.

    Fails loudly on:
      - malformed JSON lines
      - missing "text" field
      - empty text
      - non-integer repetitions
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Canary file not found: {path}")

    canaries: List[Canary] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, 1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                obj = json.loads(raw_line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Canary file {path!r} line {lineno}: invalid JSON — {e}"
                )
            if not isinstance(obj, dict):
                raise ValueError(
                    f"Canary file {path!r} line {lineno}: expected a JSON object, "
                    f"got {type(obj).__name__}"
                )
            if "text" not in obj:
                raise ValueError(
                    f"Canary file {path!r} line {lineno}: missing required field 'text'"
                )
            text = obj["text"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"Canary file {path!r} line {lineno}: 'text' must be a non-empty string"
                )

            cid = obj.get("id", f"canary_{lineno:05d}")
            if not isinstance(cid, str) or not cid.strip():
                raise ValueError(
                    f"Canary file {path!r} line {lineno}: 'id' must be a non-empty string"
                )

            # Per-canary repetition override, or expand the schedule
            if "repetitions" in obj:
                reps_val = obj["repetitions"]
                if not isinstance(reps_val, int) or reps_val < 1:
                    raise ValueError(
                        f"Canary file {path!r} line {lineno}: "
                        f"'repetitions' must be a positive integer, got {reps_val!r}"
                    )
                reps_list = [reps_val]
            else:
                reps_list = repetitions

            for reps in reps_list:
                canaries.append(Canary(id=cid, text=text, repetitions=reps))

    if not canaries:
        raise ValueError(f"Canary file {path!r} contains no valid canary entries.")

    # Validate uniqueness of (id, repetitions) pairs
    seen: Dict[Tuple[str, int], int] = {}
    for c in canaries:
        key = (c.id, c.repetitions)
        if key in seen:
            raise ValueError(
                f"Duplicate (id={c.id!r}, repetitions={c.repetitions}) in canary file."
            )
        seen[key] = 1

    return canaries


# ---------------------------------------------------------------------------
# Budget slicing
# ---------------------------------------------------------------------------

def _slice_budget(
    source_path: str,
    budget_type: str,
    budget_value: int,
) -> List[str]:
    """
    Read the first N chars or N lines from source_path.

    Returns a list of lines (without trailing newlines).
    Fails loudly if the source file is empty or unreadable.
    """
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Base corpus not found: {source_path}")

    lines: List[str] = []
    total_chars = 0

    with open(source_path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line:
                continue  # skip blank lines
            if budget_type == "raw_chars":
                if total_chars >= budget_value:
                    break
                lines.append(line)
                total_chars += len(line)
            elif budget_type == "raw_lines":
                if len(lines) >= budget_value:
                    break
                lines.append(line)
                total_chars += len(line)
            else:
                raise ValueError(f"Unknown budget_type: {budget_type!r}")

    if not lines:
        raise ValueError(
            f"Base corpus {source_path!r} yielded 0 lines under budget "
            f"{budget_type}={budget_value}."
        )

    logger.info(
        "Budget slice: %d lines, %d chars (budget_type=%s, budget_value=%d)",
        len(lines), total_chars, budget_type, budget_value,
    )
    return lines


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------

def _apply_template(template: str, canary_text: str) -> str:
    """
    Render a canary string through the injection template.

    Template uses Python str.format with the keyword {canary}.
    Examples:
      "{canary}"              -> canary text verbatim
      "SECRET: {canary} END" -> wrapped form

    Fails loudly if:
      - template does not contain the literal substring "{canary}"
      - template.format() raises (unknown fields, bad syntax)
      - the rendered result is empty or whitespace-only
    """
    if "{canary}" not in template:
        raise ValueError(
            f"Injection template {template!r} does not contain the "
            f"'{{canary}}' placeholder.  Every canary would produce the "
            f"same line, which is likely a mistake."
        )
    try:
        result = template.format(canary=canary_text)
    except (KeyError, IndexError, ValueError) as e:
        raise ValueError(
            f"Injection template {template!r} is invalid: {e}.  "
            f"Use {{canary}} as the only placeholder."
        )
    if not result.strip():
        raise ValueError(
            f"Template {template!r} + canary produced an empty string."
        )
    return result


def _inject_canaries(
    clean_lines: List[str],
    canaries: List[Canary],
    template: str,
    seed: int,
) -> Tuple[List[str], List[CanaryRecord]]:
    """
    Insert canary lines into a copy of clean_lines at deterministic positions.

    Correctness invariants (auditable):

    1. All insertion points are sampled from the CLEAN corpus index space
       range(0, n_clean + 1) ONLY.  n_clean is fixed before sampling begins
       and is never updated during sampling.  The sampling space does not grow
       as canaries are added.

    2. An insertion point p means "place this canary line immediately before
       clean_lines[p]".  p == n_clean means append after the last clean line.

    3. All insertion points for all canaries at all repetition levels are
       pre-computed and globally sorted BEFORE the output list is assembled.
       The merge pass is a single sequential scan.

    4. record_map is keyed on (id, repetition_level) so that different
       repetition levels of the same canary id remain separate records.

    Returns:
      (result_lines, records)
      result_lines : the complete D_canary line list
      records      : one CanaryRecord per (id, repetition_level) pair;
                     len(record.positions) == record.repetition_level always.
    """
    from collections import defaultdict

    rng = random.Random(seed)
    n_clean = len(clean_lines)

    # ------------------------------------------------------------------
    # Phase 1: pre-compute ALL insertion points in clean corpus space.
    #
    # For each canary with rep_level R, we spread R copies evenly across
    # [0, n_clean] using fractional offsets + small RNG jitter, then
    # convert to integers in range(0, n_clean + 1).
    #
    # All sampling is done here; n_clean is constant throughout.
    # ------------------------------------------------------------------

    # Each entry: (clean_pos, injected_text, canary_id, rep_level)
    insertions: List[Tuple[int, str, str, int]] = []

    for canary in canaries:
        injected_text = _apply_template(template, canary.text)
        rep_level = canary.repetitions
        for k in range(rep_level):
            # Evenly spaced fraction with bounded jitter, mapped to clean index space.
            frac = (k + rng.random() * 0.5) / rep_level
            frac = min(max(frac, 0.0), 1.0)
            clean_pos = min(int(frac * (n_clean + 1)), n_clean)
            insertions.append((clean_pos, injected_text, canary.id, rep_level))

    # ------------------------------------------------------------------
    # Phase 2: sort all insertions by clean position (stable sort).
    # Ties (same clean_pos) retain original canary ordering.
    # ------------------------------------------------------------------
    insertions.sort(key=lambda x: x[0])

    # ------------------------------------------------------------------
    # Phase 3: group insertions by their clean_pos for O(n) merge.
    # ------------------------------------------------------------------
    by_clean_pos: Dict[int, List[Tuple[str, str, int]]] = defaultdict(list)
    for clean_pos, itext, cid, rep_level in insertions:
        by_clean_pos[clean_pos].append((itext, cid, rep_level))

    # ------------------------------------------------------------------
    # Phase 4: single-pass merge.
    #
    # Walk clean_pos from 0 to n_clean.  At each step:
    #   a) emit all canaries scheduled before clean_lines[clean_pos]
    #   b) emit clean_lines[clean_pos] (if not past end)
    #
    # Positions recorded in each CanaryRecord are 0-based output line
    # indices in D_canary — exactly what is written to disk.
    # ------------------------------------------------------------------
    result: List[str] = []
    record_map: Dict[Tuple[str, int], CanaryRecord] = {}
    output_idx = 0

    # Build a text lookup for canary id → raw text (needed for CanaryRecord)
    id_to_text: Dict[str, str] = {c.id: c.text for c in canaries}

    for clean_pos in range(n_clean + 1):
        # Emit any canaries scheduled at this clean position
        for itext, cid, rep_level in by_clean_pos.get(clean_pos, []):
            result.append(itext)
            key = (cid, rep_level)
            if key not in record_map:
                record_map[key] = CanaryRecord(
                    id=cid,
                    text=id_to_text[cid],
                    repetition_level=rep_level,
                    injected_text=itext,
                    positions=[],
                )
            record_map[key].positions.append(output_idx)
            output_idx += 1

        # Emit the clean line (skip the sentinel step at clean_pos == n_clean)
        if clean_pos < n_clean:
            result.append(clean_lines[clean_pos])
            output_idx += 1

    # Invariant check: every record's position count must equal its rep_level
    for (cid, rep_level), rec in record_map.items():
        if len(rec.positions) != rep_level:
            raise RuntimeError(
                f"Injection invariant violated: canary id={cid!r} "
                f"rep_level={rep_level} has {len(rec.positions)} positions "
                f"(expected {rep_level}).  This is a bug."
            )

    return result, list(record_map.values())


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
# Main build function
# ---------------------------------------------------------------------------

def build_corpus(
    base_source: str,
    canary_file: str,
    repetitions: List[int],
    budget_type: str,
    budget_value: int,
    output_dir: str,
    run_id: str,
    seed: int,
    template: str = "{canary}",
) -> Manifest:
    """
    Build D_clean and D_canary and write all outputs.

    Parameters
    ----------
    base_source   : path to the raw text file
    canary_file   : path to the JSONL canary list
    repetitions   : default repetition schedule (e.g. [1, 10, 100])
    budget_type   : "raw_chars" or "raw_lines"
    budget_value  : N chars or N lines to keep from base_source
    output_dir    : root output directory (run sub-dir appended automatically)
    run_id        : experiment identifier, used as output sub-directory name
    seed          : RNG seed for deterministic injection
    template      : injection template; must contain {canary}

    Returns
    -------
    Manifest dataclass with full metadata.
    """
    run_dir = os.path.join(output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    logger.info("=== build_corpus: run_id=%s ===", run_id)
    logger.info("base_source   = %s", base_source)
    logger.info("canary_file   = %s", canary_file)
    logger.info("repetitions   = %s", repetitions)
    logger.info("budget        = %s=%d", budget_type, budget_value)
    logger.info("seed          = %d", seed)
    logger.info("template      = %r", template)
    logger.info("output_dir    = %s", run_dir)

    # 1) Validate template before anything else
    _apply_template(template, "TEST_CANARY")

    # 2) Load and validate canaries
    canaries = _load_canaries(canary_file, repetitions)
    logger.info("Loaded %d canary entries", len(canaries))

    # 3) Slice base corpus to budget
    clean_lines = _slice_budget(base_source, budget_type, budget_value)
    n_clean_lines = len(clean_lines)
    n_clean_chars = sum(len(l) for l in clean_lines)

    # 4) Write D_clean
    d_clean_path = os.path.join(run_dir, "D_clean.txt")
    with open(d_clean_path, "w", encoding="utf-8") as fh:
        for line in clean_lines:
            fh.write(line + "\n")
    logger.info("Wrote D_clean: %d lines, %d chars → %s", n_clean_lines, n_clean_chars, d_clean_path)

    # 5) Build D_canary via injection
    canary_lines, records = _inject_canaries(clean_lines, canaries, template, seed)
    n_canary_lines = len(canary_lines)
    n_canary_chars = sum(len(l) for l in canary_lines)

    # 6) Write D_canary
    d_canary_path = os.path.join(run_dir, "D_canary.txt")
    with open(d_canary_path, "w", encoding="utf-8") as fh:
        for line in canary_lines:
            fh.write(line + "\n")
    logger.info("Wrote D_canary: %d lines, %d chars → %s", n_canary_lines, n_canary_chars, d_canary_path)

    # 7) Write canaries.json
    canaries_json_path = os.path.join(run_dir, "canaries.json")
    canary_records_raw = [asdict(r) for r in records]
    with open(canaries_json_path, "w", encoding="utf-8") as fh:
        json.dump(canary_records_raw, fh, indent=2, ensure_ascii=False)
    total_injections = sum(r.repetition_level for r in records)
    logger.info(
        "Wrote canary manifest: %d (id, rep_level) records, %d total injections → %s",
        len(records), total_injections, canaries_json_path,
    )

    # 8) Build and write manifest
    manifest = Manifest(
        run_id=run_id,
        seed=seed,
        budget_type=budget_type,
        budget_value=budget_value,
        base_source=os.path.abspath(base_source),
        base_source_sha256=_sha256(base_source),
        canary_file=os.path.abspath(canary_file),
        canary_file_sha256=_sha256(canary_file),
        repetitions=repetitions,
        template=template,
        n_clean_lines=n_clean_lines,
        n_clean_chars=n_clean_chars,
        n_canary_lines=n_canary_lines,
        n_canary_chars=n_canary_chars,
        n_canaries=len(records),
        output_dir=os.path.abspath(run_dir),
    )
    manifest_path = os.path.join(run_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(asdict(manifest), fh, indent=2, ensure_ascii=False)
    logger.info("Wrote manifest → %s", manifest_path)
    logger.info("=== build_corpus done ===")

    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.build_corpus",
        description=(
            "Build D_clean and D_canary training corpora. "
            "All arguments can be supplied via --config; CLI flags override config values."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config", metavar="YAML",
        help="Path to a project YAML config file. Provides defaults for all fields.",
    )
    p.add_argument(
        "--base", metavar="FILE",
        help="Path to the raw base corpus text file (overrides config corpus.base_source).",
    )
    p.add_argument(
        "--canaries", metavar="FILE",
        help="Path to the JSONL canary list (overrides config corpus.canary.file).",
    )
    p.add_argument(
        "--reps", metavar="N", nargs="+", type=int,
        help="Repetition schedule, e.g. --reps 1 10 100 (overrides config).",
    )
    p.add_argument(
        "--budget-type", choices=["raw_chars", "raw_lines"],
        help="Budget type (overrides config corpus.budget_type).",
    )
    p.add_argument(
        "--budget-value", type=int, metavar="N",
        help="Budget value (overrides config corpus.budget_value).",
    )
    p.add_argument(
        "--template", default="{canary}",
        help=(
            "Injection template. Must contain {canary}. "
            "Example: 'The secret is: {canary}'"
        ),
    )
    p.add_argument(
        "--run-id", metavar="ID",
        help="Run identifier used as output sub-directory name (overrides config run_id).",
    )
    p.add_argument(
        "--output", metavar="DIR",
        help="Root output directory (overrides config paths.corpus_dir).",
    )
    p.add_argument(
        "--seed", type=int,
        help="RNG seed for deterministic injection (overrides config seed).",
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
    base_source = canary_file = budget_type = output_dir = run_id = None
    budget_value = seed = None
    repetitions: Optional[List[int]] = None

    if args.config:
        # Import here so the module is importable without config present
        from src.config import load_config  # type: ignore[import]
        cfg = load_config(args.config)
        base_source   = cfg.corpus.base_source
        canary_file   = cfg.corpus.canary.file
        repetitions   = cfg.corpus.canary.repetitions
        budget_type   = cfg.corpus.budget_type
        budget_value  = cfg.corpus.budget_value
        output_dir    = cfg.paths.corpus_dir
        run_id        = cfg.run_id
        seed          = cfg.seed

    # CLI overrides
    if args.base:         base_source  = args.base
    if args.canaries:     canary_file  = args.canaries
    if args.reps:         repetitions  = args.reps
    if args.budget_type:  budget_type  = args.budget_type
    if args.budget_value: budget_value = args.budget_value
    if args.output:       output_dir   = args.output
    if args.run_id:       run_id       = args.run_id
    if args.seed is not None: seed     = args.seed

    # Validate all required fields are present
    missing = [
        name for name, val in [
            ("--base / corpus.base_source",          base_source),
            ("--canaries / corpus.canary.file",       canary_file),
            ("--reps / corpus.canary.repetitions",    repetitions),
            ("--budget-type / corpus.budget_type",    budget_type),
            ("--budget-value / corpus.budget_value",  budget_value),
            ("--output / paths.corpus_dir",           output_dir),
            ("--run-id / run_id",                     run_id),
            ("--seed / seed",                         seed),
        ]
        if val is None
    ]
    if missing:
        parser.error(
            f"Missing required values (provide via --config or explicit flags):\n  "
            + "\n  ".join(missing)
        )

    try:
        manifest = build_corpus(
            base_source=base_source,        # type: ignore[arg-type]
            canary_file=canary_file,        # type: ignore[arg-type]
            repetitions=repetitions,        # type: ignore[arg-type]
            budget_type=budget_type,        # type: ignore[arg-type]
            budget_value=budget_value,      # type: ignore[arg-type]
            output_dir=output_dir,          # type: ignore[arg-type]
            run_id=run_id,                  # type: ignore[arg-type]
            seed=seed,                      # type: ignore[arg-type]
            template=args.template,
        )
    except (FileNotFoundError, ValueError) as e:
        logger.error("FATAL: %s", e)
        sys.exit(1)

    print(f"\nDone.  Outputs in: {manifest.output_dir}")
    print(f"  D_clean  : {manifest.n_clean_lines} lines, {manifest.n_clean_chars} chars")
    print(f"  D_canary : {manifest.n_canary_lines} lines, {manifest.n_canary_chars} chars")
    print(f"  Canaries : {manifest.n_canaries} (id, repetition_level) records in canaries.json")


if __name__ == "__main__":
    main()
