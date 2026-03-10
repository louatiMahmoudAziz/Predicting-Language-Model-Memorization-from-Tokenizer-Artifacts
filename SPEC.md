# SPEC.md — Tokenizer-only Prediction of String-level Memorization Risk

This document is the **source of truth** for definitions, threat model, labels, evaluation rules, and smoke-test pass/fail criteria.  
If any code, config, or prompt conflicts with this file, **SPEC.md wins**.

---

## 1) Threat model (tokenizer-only auditor)

At prediction time, the auditor has access to:
- Tokenizer artifacts: vocabulary + merges (BPE) or SentencePiece `.model` (unigram), plus tokenizer normalization rules.
- A candidate string `s`.

The auditor has **no access** to:
- tokenizer training corpus,
- LM weights, gradients,
- model outputs / logprobs / generation API.

The auditor outputs a **risk score** used to prioritize strings for downstream privacy attacks (membership inference or extraction) studied in prior work.

---

## 2) Core definitions

### 2.1 Tokenizer-normalized string
For any tokenizer `Tok` and input string `s`, define:

- `s_norm = Normalize_Tok(s)` using the tokenizer’s **built-in normalizer**.

**Rule:** The *same* `s_norm` must be used for:
- tokenization,
- computing character length,
- scoring BPC.

**No fake normalization:** If tokenizer normalization cannot be reproduced programmatically, code must **fail loudly** with an actionable error.

### 2.2 Character length
Define:

- `len_chars(s) = len(s_norm)` where `s_norm = Normalize_Tok(s)`.

This is the denominator for BPC.

### 2.3 Teacher-forced total log-likelihood in bits
Let tokens be `x_1..x_T = Tok.encode(s_norm)`.

Define total negative log-likelihood (bits) under model `M` using teacher forcing:

- `total_bits_M(s) = - sum_{t=1..T-1} log2 p_M(x_{t+1} | x_{<=t})`

**Important:**
- This is **total**, not averaged per token.
- Off-by-one must be correct: predict token `t+1` from prefix up to `t`.
- BOS/EOS handling must be explicit and documented. Default: **do not add special BOS/EOS unless the tokenizer/model requires it**; score exactly what `Tok.encode(s_norm)` returns.

### 2.4 Bits-per-character (BPC)
Define:

- `BPC_M(s) = total_bits_M(s) / len_chars(s)`

If `len_chars(s) == 0` after normalization, scoring must **fail loudly** or return a clearly marked invalid value (never silently divide by zero).

**Tokenizer comparability:**  BPC is expressed in a tokenizer-comparable unit because it normalizes sequence probability by normalized character length, unlike bits-per-token which is inherently tokenizer-dependent.

(Optional robustness check: bits-per-byte; but BPC is the canonical metric.)

---

## 3) Matched target/reference models (per tokenizer)

For each tokenizer condition `Tok`, we must train **two models using the same tokenizer**:

- `M_ref^Tok`: trained on a **canary-free** corpus `D_clean`
- `M_target^Tok`: trained on a corpus `D_canary` (same base text + injected canaries)

**Rule:** A reference model must use the **exact same tokenizer** as the target model.  
No cross-tokenizer reference models are allowed.

### 3.1 Label definition (memorization proxy)
For each string `s`:

- `L^Tok(s) = ΔBPC^Tok(s) = BPC_{M_ref^Tok}(s) - BPC_{M_target^Tok}(s)`

Interpretation:
- `ΔBPC > 0` means the target assigns higher probability per character than the reference (memorization signal).

**Terminology rule:** ΔBPC is a **memorization proxy**, not guaranteed extractability.

---

## 4) Fixed budget fairness (training)

Training budgets must be **comparable across tokenizers**.

### 4.1 Fixed raw-text budget (required)
All experimental conditions must use a fixed **raw text budget**, not “same epochs”.

Allowed budget types:
- `raw_chars`: use first `N` characters from the base corpus (after the chosen corpus preprocessing), then inject canaries
- `raw_lines`: use first `N` lines

The training script must log:
- raw chars processed
- raw lines processed
- total tokens processed (tokenized)
- update steps performed

**Rule:** Do not claim fairness unless raw-text budget is fixed and logged.

---

## 5) Candidate strings

Candidate set must include diverse categories to avoid “randomness detector” artifacts:

Required buckets:
1) random canaries (UUID/base64-like)
2) structured synthetic secrets (API-key patterns, emails, SSN-like patterns)
3) rare natural phrases/entities (public, non-sensitive)
4) near-duplicates (edit distance 1–2 variants)
5) benign negatives sampled from held-out corpus distribution

No real PII should be used.

---

## 6) Tokenizer-only features (pre-training)

All features must be computed using only:
- tokenizer artifacts
- the string `s` (and its tokenizer-normalized form)

No model weights, no logprobs, no corpus-derived frequencies at prediction time.

Required baseline features:
- `len_chars(s)` (using target tokenizer normalization)
- `char_entropy(s_norm)` (Shannon entropy over characters)
- `n_tokens_target = |Tok_target.encode(s_norm)|`

Required artifact features:
- token vocab rank percentile stats (mean/min/max over tokens)
- BPE: merge-rank stats (mean/max, based on merge order)
- Unigram: piece score/logprob stats (mean/min/max from `.model`)

Required compression discrepancy:
- Choose one or more public reference tokenizers `Tok_ref`.
- Compute `n_tokens_ref = |Tok_ref.encode(Normalize_ref(s))|`
- `ΔTok(s) = n_tokens_ref - n_tokens_target`

**Rule:** reference-tokenizer normalization and target-tokenizer normalization may differ; both must be applied consistently.

---

## 7) Predictor evaluation protocol (avoid p-hacking)

### 7.1 Train/val/test split (required)
All predictive modeling must use a fixed split:
- train
- validation (for hyperparameters, thresholds)
- test (final reporting only)

Do not tune thresholds/features on test.

### 7.2 Primary metrics (tail-first)
Report:
- Ranking: precision@K, recall@K, NDCG@K (K ∈ {50, 100, 500})
- Tail: TPR@FPR≤1% (always)

### 7.3 0.1% FPR resolvability rule
Only report TPR@FPR≤0.1% if the negative set size is sufficient.

**Rule:** If `#negatives < 100000`, then:
- do not report 0.1% as a headline metric,
- mark it as “not resolvable” and use 1% FPR instead.

AUROC/AUPRC are secondary summary metrics, not primary.

---

## 8) Extraction validation (secondary, non-circular)

Extraction is optional and secondary. If performed:

**Non-circularity rule:** Select strings for extraction **by predicted tokenizer-only risk score**, not by true ΔBPC label.

Report extraction outcomes as confirmation that high predicted memorization risk can translate to discoverable leakage.

---

## 9) Colab smoke tests (mandatory gate before HPC)

### Test 1 — “Nuke” plumbing unit test
Goal: Verify the end-to-end math and matching works.

Setup:
- tiny corpus (1–5 MB)
- 5–10 canaries
- high repetition: 200×–1000×
- train matched ref/target quickly

PASS if:
- repeated canaries show clearly positive ΔBPC
- benign controls have ΔBPC near 0

If FAIL: fix code/math before proceeding.

### Test 2 — Mini-realistic sanity check
Goal: Verify memorization signal exists in a non-degenerate regime.

Setup:
- WikiText subset (or similar)
- repetition schedule includes 1×, 10×, 100×
- enough training to learn (loss decreases; reasonable held-out perplexity)

PASS if:
- ΔBPC increases with repetition on average
- ΔBPC has variance across strings at the same repetition level
- a trivial baseline (e.g., token count) achieves better-than-random top-K retrieval

If Test 1 passes but Test 2 fails:
- not a code bug; adjust model size/budget/repetition upper bound/hyperparameters.

**Gate rule:** Do not start HPC sweeps until Test 2 passes.

---

## 10) Logging and failure rules (no silent failures)

- Any mismatch between:
  - normalized string used for tokenization
  - normalized string used for character length
  must raise or be explicitly logged as a fatal error.

- Any ref/target mismatch (wrong tokenizer, wrong run_id, misaligned ids) must fail loudly.

- Any unresolvable metric (0.1% FPR with too few negatives) must be explicitly flagged.

- All artifacts must be written in Parquet where appropriate (features, labels) to avoid CSV I/O issues at scale.

---