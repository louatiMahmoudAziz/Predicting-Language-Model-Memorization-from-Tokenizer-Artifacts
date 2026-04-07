#!/usr/bin/env python3
"""
gen_pile_candidates.py
----------------------
Generate candidate strings for pretrained-model memorization evaluation.

Samples sequences from The Pile (Pythia's training data) at varying
expected memorization levels, plus out-of-distribution negatives.

Candidate categories:
  1. High-frequency Pile sequences (likely memorized)
  2. Medium-frequency Pile sequences
  3. Low-frequency / unique Pile sequences
  4. Synthetic negatives NOT in The Pile (should NOT be memorized)
  5. Common knowledge strings (ambiguous — memorized from many sources)

Usage:
    python scripts/gen_pile_candidates.py [--n-per-bucket 500] [--output data/candidates/pile_candidates.jsonl]
"""

import argparse
import hashlib
import json
import logging
import os
import random
import string
import sys

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Synthetic negatives — strings very unlikely to be in any training corpus
# ---------------------------------------------------------------------------

def _generate_synthetic_negatives(n: int, seed: int = 42) -> list:
    """Generate strings that should NOT appear in The Pile."""
    rng = random.Random(seed)
    candidates = []

    # Random alphanumeric strings
    for i in range(n // 4):
        length = rng.randint(50, 200)
        text = "".join(rng.choices(string.ascii_letters + string.digits + " ", k=length))
        candidates.append({
            "id": f"synthetic_random_{i:04d}",
            "text": text,
            "category": "synthetic_random",
            "expected_memorization": "none",
        })

    # Shuffled real sentences (destroy meaning while keeping character distribution)
    templates = [
        "The {adj} {noun} {verb} {adv} across the {noun2} while the {noun3} watched carefully.",
        "In {year}, Professor {name} discovered that {noun} could {verb} without any {noun2}.",
        "According to internal document {code}, the {noun} division reported {number}% growth in {noun2}.",
        "{name} sent an email to {name2} about the {adj} {noun} project deadline on {date}.",
    ]
    adjs = ["quantum", "bilateral", "recursive", "stochastic", "ergodic", "holomorphic",
            "anisotropic", "contravariant", "endomorphic", "diffeomorphic"]
    nouns = ["algorithm", "manifold", "eigenvalue", "trajectory", "isomorphism",
             "deformation", "cohomology", "functor", "sheaf", "fibration"]
    verbs = ["converges", "bifurcates", "interpolates", "oscillates", "propagates"]
    names = ["Karpinsky", "Thirumalai", "Okonkwo", "Reinhardtsen", "Matsuzaki"]
    adverbs = ["asymptotically", "monotonically", "holomorphically", "ergodically"]

    for i in range(n // 4):
        template = rng.choice(templates)
        text = template.format(
            adj=rng.choice(adjs), noun=rng.choice(nouns), verb=rng.choice(verbs),
            adv=rng.choice(adverbs), noun2=rng.choice(nouns), noun3=rng.choice(nouns),
            year=rng.randint(2035, 2050), name=rng.choice(names), name2=rng.choice(names),
            code=f"INT-{rng.randint(10000,99999)}", number=rng.randint(10,99),
            date=f"2040-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
        )
        candidates.append({
            "id": f"synthetic_template_{i:04d}",
            "text": text,
            "category": "synthetic_template",
            "expected_memorization": "none",
        })

    # UUID-like strings
    for i in range(n // 4):
        parts = [
            "".join(rng.choices("0123456789abcdef", k=8)),
            "".join(rng.choices("0123456789abcdef", k=4)),
            "".join(rng.choices("0123456789abcdef", k=4)),
            "".join(rng.choices("0123456789abcdef", k=4)),
            "".join(rng.choices("0123456789abcdef", k=12)),
        ]
        uuid = "-".join(parts)
        text = f"Transaction ID: {uuid} | Amount: ${rng.randint(100,99999)}.{rng.randint(10,99)} | Status: {'completed' if rng.random() > 0.3 else 'pending'}"
        candidates.append({
            "id": f"synthetic_uuid_{i:04d}",
            "text": text,
            "category": "synthetic_uuid",
            "expected_memorization": "none",
        })

    # Future-dated content (definitely not in pre-2023 training data)
    for i in range(n - len(candidates)):
        year = rng.randint(2035, 2060)
        text = f"In the fiscal year {year}, the global {rng.choice(nouns)} index reached {rng.randint(1000,9999)}.{rng.randint(10,99)} points, representing a {rng.choice(['significant', 'moderate', 'unexpected'])} shift from previous {rng.choice(nouns)} projections."
        candidates.append({
            "id": f"synthetic_future_{i:04d}",
            "text": text,
            "category": "synthetic_future",
            "expected_memorization": "none",
        })

    return candidates


# ---------------------------------------------------------------------------
# Known-memorizable content (high likelihood of appearing in The Pile)
# ---------------------------------------------------------------------------

def _generate_known_content(n: int, seed: int = 42) -> list:
    """
    Generate strings that are VERY LIKELY to appear verbatim in The Pile.

    These are famous quotes, license headers, standard boilerplate text, etc.
    that appear thousands of times across the internet.
    """
    rng = random.Random(seed)
    candidates = []

    # Software license headers (appear thousands of times in code repos)
    license_texts = [
        "Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files",
        "Licensed under the Apache License, Version 2.0 (the \"License\"); you may not use this file except in compliance with the License.",
        "This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License",
        "Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met",
        "THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS \"AS IS\" AND ANY EXPRESS OR IMPLIED WARRANTIES",
        "Copyright (c) 2015 The authors of the project. All rights reserved. Use of this source code is governed by a BSD-style license",
    ]

    for i, text in enumerate(license_texts):
        candidates.append({
            "id": f"license_{i:04d}",
            "text": text,
            "category": "license_boilerplate",
            "expected_memorization": "high",
        })

    # HTML boilerplate (extremely common)
    html_texts = [
        '<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">',
        '<div class="container"><div class="row"><div class="col-md-12"><h1>Welcome</h1></div></div></div>',
        'function(e){e.preventDefault();var t=document.getElementById',
    ]

    for i, text in enumerate(html_texts):
        candidates.append({
            "id": f"html_{i:04d}",
            "text": text,
            "category": "html_boilerplate",
            "expected_memorization": "high",
        })

    # Python boilerplate (extremely common in GitHub)
    python_texts = [
        "if __name__ == '__main__':\n    import argparse\n    parser = argparse.ArgumentParser(description=",
        "def __init__(self, *args, **kwargs):\n        super().__init__(*args, **kwargs)",
        "import os\nimport sys\nimport json\nimport logging\nfrom typing import Any, Dict, List, Optional, Tuple",
        "class Config:\n    def __init__(self, config_path: str):\n        with open(config_path, 'r') as f:\n            self.config = json.load(f)",
    ]

    for i, text in enumerate(python_texts):
        candidates.append({
            "id": f"python_{i:04d}",
            "text": text,
            "category": "code_boilerplate",
            "expected_memorization": "high",
        })

    # Wikipedia-style common knowledge (likely in Pile via Wikipedia dump)
    wiki_texts = [
        "The United States Declaration of Independence was adopted by the Continental Congress on July 4, 1776. It announced that the thirteen American colonies",
        "Deoxyribonucleic acid is a molecule composed of two polynucleotide chains that coil around each other to form a double helix",
        "The speed of light in vacuum, commonly denoted c, is a universal physical constant that is exactly equal to 299,792,458 metres per second",
        "Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience",
        "The mitochondria is the powerhouse of the cell. It is responsible for producing adenosine triphosphate (ATP), the energy currency of the cell.",
    ]

    for i, text in enumerate(wiki_texts):
        candidates.append({
            "id": f"wiki_{i:04d}",
            "text": text,
            "category": "wikipedia_common",
            "expected_memorization": "high",
        })

    # Pad to requested count with variations
    while len(candidates) < n:
        idx = len(candidates)
        base = rng.choice(license_texts + python_texts + wiki_texts)
        # Add slight variations
        text = base + f" [{rng.randint(1,999)}]"
        candidates.append({
            "id": f"known_var_{idx:04d}",
            "text": text,
            "category": "known_variant",
            "expected_memorization": "medium",
        })

    return candidates[:n]


# ---------------------------------------------------------------------------
# Pile-sampled content (requires download)
# ---------------------------------------------------------------------------

def _sample_from_pile(n: int, seed: int = 42) -> list:
    """
    Sample sequences from The Pile dataset via HuggingFace datasets.

    Falls back to synthetic content if The Pile is not available.
    """
    candidates = []
    rng = random.Random(seed)

    try:
        from datasets import load_dataset
        logger.info("Loading Pile validation split from HuggingFace...")

        # Use the deduped Pile val set (smaller, manageable download)
        ds = load_dataset(
            "EleutherAI/the_pile_deduplicated",
            split="validation",
            streaming=True,
        )

        # Sample diverse sequences
        buffer = []
        for i, example in enumerate(ds):
            text = example.get("text", "")
            if 50 <= len(text) <= 500:
                buffer.append(text)
            if len(buffer) >= n * 5:
                break
            if i > 100000:
                break

        # Random sample from buffer
        if buffer:
            selected = rng.sample(buffer, min(n, len(buffer)))
            for i, text in enumerate(selected):
                candidates.append({
                    "id": f"pile_sample_{i:04d}",
                    "text": text.strip(),
                    "category": "pile_sample",
                    "expected_memorization": "unknown",
                })
            logger.info("Sampled %d sequences from The Pile", len(candidates))

    except Exception as e:
        logger.warning("Could not load The Pile: %s. Using WikiText-103 instead.", e)

    # Fallback: use WikiText-103 (known to be in Pythia's training data)
    if len(candidates) < n:
        try:
            from datasets import load_dataset
            logger.info("Falling back to WikiText-103...")

            ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
            texts = [t for t in ds["text"] if 50 <= len(t.strip()) <= 500 and t.strip()]
            selected = rng.sample(texts, min(n - len(candidates), len(texts)))

            for i, text in enumerate(selected):
                candidates.append({
                    "id": f"wikitext_sample_{i:04d}",
                    "text": text.strip(),
                    "category": "wikitext_sample",
                    "expected_memorization": "unknown",
                })
            logger.info("Sampled %d sequences from WikiText-103", len(selected))

        except Exception as e:
            logger.warning("WikiText-103 also unavailable: %s", e)

    return candidates[:n]


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_pile_candidates(
    n_per_bucket: int = 500,
    output_path: str = "data/candidates/pile_candidates.jsonl",
    seed: int = 42,
) -> str:
    """
    Generate the full candidate set for pretrained model evaluation.

    Buckets:
      1. Known-memorizable content (license headers, boilerplate, wiki)
      2. Pile-sampled content (ground truth training data)
      3. Synthetic negatives (should NOT be memorized)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    logger.info("Generating pile candidates: %d per bucket", n_per_bucket)

    # Generate each bucket
    known = _generate_known_content(n_per_bucket, seed=seed)
    pile_samples = _sample_from_pile(n_per_bucket, seed=seed + 1)
    negatives = _generate_synthetic_negatives(n_per_bucket, seed=seed + 2)

    all_candidates = known + pile_samples + negatives

    # Deduplicate by text hash
    seen_hashes = set()
    deduped = []
    for c in all_candidates:
        h = hashlib.sha256(c["text"].encode()).hexdigest()[:16]
        if h not in seen_hashes:
            seen_hashes.add(h)
            deduped.append(c)

    logger.info(
        "Total candidates: %d (known=%d, pile=%d, negatives=%d, after dedup=%d)",
        len(all_candidates), len(known), len(pile_samples), len(negatives), len(deduped),
    )

    # Write JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for c in deduped:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    logger.info("Candidates written to: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate candidates for pretrained model evaluation")
    parser.add_argument("--n-per-bucket", type=int, default=500)
    parser.add_argument("--output", default="data/candidates/pile_candidates.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    generate_pile_candidates(
        n_per_bucket=args.n_per_bucket,
        output_path=args.output,
        seed=args.seed,
    )
