"""
Generate realistic experiment data for configs/colab_real.yaml.

Outputs:
    data/raw/wikitext103.txt           ~180 MB  (WikiText-103-raw train split)
    data/candidates/canaries_real.jsonl          (100 synthetic canary strings)
    data/candidates/candidates_real.jsonl        (100 canaries + 900 benign strings)

Usage:
    pip install datasets          # only needed for this script
    python scripts/gen_real_data.py
"""
from __future__ import annotations

import json
import os
import random
import sys


# ---------------------------------------------------------------------------
# 1. Download WikiText-103-raw and save as plain text
# ---------------------------------------------------------------------------

def download_corpus(out_path: str) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "ERROR: 'datasets' package not found.\n"
            "  pip install datasets\n"
            "Then re-run this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    if os.path.isfile(out_path) and os.path.getsize(out_path) > 50_000_000:
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"corpus already exists: {out_path}  ({size_mb:.1f} MB) — skipping download")
        return

    print("Downloading WikiText-103-raw-v1 (train split) ...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n_lines = 0
    n_chars = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            line = row["text"]
            if line.strip():
                f.write(line.rstrip("\n") + "\n")
                n_lines += 1
                n_chars += len(line)

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"corpus: {n_lines:,} lines, {n_chars:,} chars, {size_mb:.1f} MB -> {out_path}")


# ---------------------------------------------------------------------------
# 2. Generate synthetic canary strings
# ---------------------------------------------------------------------------

CANARY_TEMPLATES = [
    "sk-proj-{word}_{hex:016x}",
    "AKIA{word}{hex:08X}",
    "ghp_{word}_{hex:012x}_{word2}",
    "xoxb-{dec}-{word}-{hex:08x}",
    "-----BEGIN RSA KEY {word} {hex:016X}-----",
    "bearer {word}.{hex:012x}.{word2}",
    "password={word}_{dec}_{hex:08x}",
    "ssn:{dec3}-{dec2}-{dec4}",
    "credit_card:{dec16}",
    "email:user.{word}@corp-{hex:04x}.internal",
]


def _make_canary(rng: random.Random, template: str) -> str:
    word = "".join(rng.choices("abcdefghjkmnpqrstuvwxyz", k=rng.randint(4, 8)))
    word2 = "".join(rng.choices("ABCDEFGHJKLMNPQRSTUVWXYZ", k=rng.randint(3, 6)))
    hex_val = rng.getrandbits(64)
    dec = str(rng.randint(100000, 999999))
    dec2 = str(rng.randint(10, 99))
    dec3 = str(rng.randint(100, 999))
    dec4 = str(rng.randint(1000, 9999))
    dec16 = "".join(str(rng.randint(0, 9)) for _ in range(16))

    s = template
    s = s.replace("{word}", word)
    s = s.replace("{word2}", word2)
    s = s.replace("{dec}", dec)
    s = s.replace("{dec2}", dec2)
    s = s.replace("{dec3}", dec3)
    s = s.replace("{dec4}", dec4)
    s = s.replace("{dec16}", dec16)

    # Handle format-style hex placeholders
    import re
    def _fmt_hex(m: "re.Match") -> str:
        spec = m.group(1)
        return format(hex_val & ((1 << (len(spec.rstrip("xX")) * 4)) - 1 | 1), spec)
    s = re.sub(r"\{hex:([0-9]+[xX])\}", _fmt_hex, s)

    return s


def generate_canaries(out_path: str, n_canaries: int = 100, seed: int = 99) -> list:
    rng = random.Random(seed)
    reps_schedule = [1, 3, 10, 50, 100]

    entries = []
    idx = 1
    per_level = n_canaries // len(reps_schedule)
    for reps in reps_schedule:
        for _ in range(per_level):
            tmpl = rng.choice(CANARY_TEMPLATES)
            text = _make_canary(rng, tmpl)
            entries.append({
                "id": f"canary_{idx:04d}",
                "text": text,
                "repetitions": reps,
            })
            idx += 1

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    print(f"canaries: {len(entries)} entries -> {out_path}")
    for reps in reps_schedule:
        n = sum(1 for e in entries if e["repetitions"] == reps)
        print(f"  rep={reps:>4d}: {n} canaries")

    return entries


# ---------------------------------------------------------------------------
# 3. Sample benign strings from corpus and create combined candidates file
# ---------------------------------------------------------------------------

def generate_candidates(
    corpus_path: str,
    canary_entries: list,
    out_path: str,
    n_benign: int = 900,
    seed: int = 77,
) -> None:
    rng = random.Random(seed)

    print(f"Sampling {n_benign} benign strings from corpus ...")
    lines = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if len(stripped) >= 30 and len(stripped) <= 500:
                lines.append(stripped)

    if len(lines) < n_benign:
        raise ValueError(
            f"Corpus has only {len(lines)} lines in [30..500] char range, "
            f"need {n_benign}. Use a larger corpus."
        )

    sampled = rng.sample(lines, n_benign)

    # Canary entries for the candidates file (without repetitions field)
    candidates = []
    for e in canary_entries:
        candidates.append({"id": e["id"], "text": e["text"]})

    for i, text in enumerate(sampled, 1):
        candidates.append({"id": f"benign_{i:04d}", "text": text})

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c) + "\n")

    n_canary = len(canary_entries)
    print(
        f"candidates: {len(candidates)} total "
        f"({n_canary} canaries + {n_benign} benign) -> {out_path}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    corpus_path = "data/raw/wikitext103.txt"
    canary_path = "data/candidates/canaries_real.jsonl"
    candidates_path = "data/candidates/candidates_real.jsonl"

    download_corpus(corpus_path)
    canary_entries = generate_canaries(canary_path, n_canaries=100, seed=99)
    generate_candidates(corpus_path, canary_entries, candidates_path, n_benign=900, seed=77)

    print("\nDone. Ready to run:")
    print("  python -m src.run_pipeline --config configs/colab_real.yaml")
