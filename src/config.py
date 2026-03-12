"""
config.py
---------
Loads and validates YAML configuration files for the memorization risk pipeline.

Schema matches SPEC.md:
  - Transformer LMs trained from scratch (ref + target per tokenizer)
  - Fixed raw-text budget for training fairness
  - Corpus with D_clean/D_canary + canary repetition schedule
  - BPC scoring from transformer logprobs with tokenizer-normalized strings
  - Predictor with explicit train/val/test split
  - Evaluation with tail-first metrics + 0.1% FPR resolvability rule
  - Optional KenLM baseline (OFF by default)

Usage:
    from src.config import load_config
    cfg = load_config("configs/colab_mini.yaml")

    cfg.run_id
    cfg.seed
    cfg.paths.ref_model_dir
    cfg.corpus.budget_type
    cfg.tokenizer.vocab_size
    cfg.lm.d_model
    cfg.lm.training.max_steps
    cfg.scoring.metric
    cfg.predictor.split.test
    cfg.evaluation.fpr_0_1_min_negatives
"""

from __future__ import annotations

import os
import yaml
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Sub-config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PathsConfig:
    raw_dir: str
    corpus_dir: str
    candidates_dir: str
    tokenizer_dir: str
    ref_model_dir: str
    target_model_dir: str
    features_dir: str
    labels_dir: str
    results_dir: str


@dataclass
class CanaryConfig:
    file: str
    repetitions: List[int]


@dataclass
class CorpusConfig:
    base_source: str
    budget_type: str            # raw_chars | raw_lines
    budget_value: int
    canary: CanaryConfig
    candidates_file: str = ""   # separate file for scoring/features; if empty, falls back to canary.file


@dataclass
class TokenizerConfig:
    type: str                   # bpe | unigram
    vocab_size: int
    min_frequency: int
    special_tokens: List[str] = field(default_factory=list)


@dataclass
class LMTrainingConfig:
    batch_size: int
    learning_rate: float
    max_steps: int
    warmup_steps: int
    weight_decay: float
    log_every: int
    eval_every: int
    checkpoint_every: int


@dataclass
class BaselineKenLMConfig:
    enabled: bool
    order: int
    smoothing: str


@dataclass
class LMConfig:
    architecture: str           # transformer
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    dropout: float
    max_seq_len: int
    training: LMTrainingConfig
    baseline_kenlm: BaselineKenLMConfig


@dataclass
class ScoringConfig:
    metric: str                 # bpc
    batch_size: int
    add_bos: bool
    add_eos: bool
    allow_truncation: bool = False


@dataclass
class SplitConfig:
    train: float
    val: float
    test: float


@dataclass
class PredictorConfig:
    model_type: str             # logistic | xgboost | rf
    max_iter: int
    n_estimators: int
    split: SplitConfig
    threshold: float


@dataclass
class EvaluationConfig:
    ranking_k: List[int]
    tail_fpr: List[float]
    fpr_0_1_min_negatives: int
    secondary_metrics: List[str]


@dataclass
class Config:
    run_id: str
    seed: int
    paths: PathsConfig
    corpus: CorpusConfig
    tokenizer: TokenizerConfig
    lm: LMConfig
    scoring: ScoringConfig
    predictor: PredictorConfig
    evaluation: EvaluationConfig

    def __repr__(self) -> str:
        lines = [f"Config(run_id={self.run_id!r}, seed={self.seed})"]
        lines.append(f"  corpus.budget            = {self.corpus.budget_value} {self.corpus.budget_type}")
        lines.append(f"  corpus.canary_reps       = {self.corpus.canary.repetitions}")
        lines.append(f"  tokenizer.type           = {self.tokenizer.type}")
        lines.append(f"  tokenizer.vocab_size     = {self.tokenizer.vocab_size}")
        lines.append(f"  lm.architecture          = {self.lm.architecture}")
        lines.append(f"  lm.d_model               = {self.lm.d_model}")
        lines.append(f"  lm.n_layers              = {self.lm.n_layers}")
        lines.append(f"  lm.training.max_steps    = {self.lm.training.max_steps}")
        lines.append(f"  lm.baseline_kenlm        = {'ON' if self.lm.baseline_kenlm.enabled else 'OFF'}")
        lines.append(f"  scoring.metric           = {self.scoring.metric}")
        lines.append(f"  predictor.model_type     = {self.predictor.model_type}")
        lines.append(f"  predictor.split          = {self.predictor.split.train}/{self.predictor.split.val}/{self.predictor.split.test}")
        lines.append(f"  evaluation.ranking_k     = {self.evaluation.ranking_k}")
        lines.append(f"  evaluation.tail_fpr      = {self.evaluation.tail_fpr}")
        lines.append(f"  evaluation.fpr_0_1_min_n = {self.evaluation.fpr_0_1_min_negatives}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_REQUIRED_TOP = {"run_id", "seed", "paths", "corpus", "tokenizer", "lm", "scoring", "predictor", "evaluation"}

_REQUIRED_PATHS = {
    "raw_dir", "corpus_dir", "candidates_dir", "tokenizer_dir",
    "ref_model_dir", "target_model_dir",
    "features_dir", "labels_dir", "results_dir",
}

_REQUIRED_CORPUS = {"base_source", "budget_type", "budget_value", "canary"}
_REQUIRED_CANARY = {"file", "repetitions"}

_REQUIRED_TOKENIZER = {"type", "vocab_size", "min_frequency"}

_REQUIRED_LM = {"architecture", "d_model", "n_heads", "n_layers", "d_ff", "dropout", "max_seq_len", "training"}
_REQUIRED_LM_TRAINING = {"batch_size", "learning_rate", "max_steps", "warmup_steps", "weight_decay",
                          "log_every", "eval_every", "checkpoint_every"}

_REQUIRED_SCORING = {"metric", "batch_size", "add_bos", "add_eos"}

_REQUIRED_PREDICTOR = {"model_type", "max_iter", "n_estimators", "split", "threshold"}
_REQUIRED_SPLIT = {"train", "val", "test"}

_REQUIRED_EVALUATION = {"ranking_k", "tail_fpr", "fpr_0_1_min_negatives", "secondary_metrics"}

_VALID_TOKENIZER_TYPES = {"bpe", "unigram"}
_VALID_BUDGET_TYPES = {"raw_chars", "raw_lines"}
_VALID_LM_ARCHITECTURES = {"transformer"}
_VALID_SCORING_METRICS = {"bpc"}
_VALID_PREDICTOR_TYPES = {"logistic", "xgboost", "rf"}


def _check_keys(d: dict, required: set, section: str) -> None:
    missing = required - d.keys()
    if missing:
        raise KeyError(f"Config [{section}] missing required keys: {sorted(missing)}")


def _check_enum(value: str, allowed: set, field_name: str) -> None:
    if value not in allowed:
        raise ValueError(f"{field_name}={value!r} not in {sorted(allowed)}")


def _validate(raw: dict) -> None:
    """Run all structural and semantic validation. Raises on any problem."""
    _check_keys(raw, _REQUIRED_TOP, "root")

    # Reject old-style configs that have the deprecated ngram/kenlm-only lm section
    lm = raw["lm"]
    if isinstance(lm, dict) and "order" in lm and "architecture" not in lm:
        raise ValueError(
            "Config uses deprecated ngram/kenlm LM schema. "
            "SPEC.md requires transformer LMs. Update your config."
        )

    _check_keys(raw["paths"], _REQUIRED_PATHS, "paths")
    _check_keys(raw["corpus"], _REQUIRED_CORPUS, "corpus")
    _check_keys(raw["corpus"]["canary"], _REQUIRED_CANARY, "corpus.canary")
    _check_keys(raw["tokenizer"], _REQUIRED_TOKENIZER, "tokenizer")
    _check_keys(raw["lm"], _REQUIRED_LM, "lm")
    _check_keys(raw["lm"]["training"], _REQUIRED_LM_TRAINING, "lm.training")
    _check_keys(raw["scoring"], _REQUIRED_SCORING, "scoring")
    _check_keys(raw["predictor"], _REQUIRED_PREDICTOR, "predictor")
    _check_keys(raw["predictor"]["split"], _REQUIRED_SPLIT, "predictor.split")
    _check_keys(raw["evaluation"], _REQUIRED_EVALUATION, "evaluation")

    _check_enum(raw["tokenizer"]["type"], _VALID_TOKENIZER_TYPES, "tokenizer.type")
    _check_enum(raw["corpus"]["budget_type"], _VALID_BUDGET_TYPES, "corpus.budget_type")
    _check_enum(raw["lm"]["architecture"], _VALID_LM_ARCHITECTURES, "lm.architecture")
    _check_enum(raw["scoring"]["metric"], _VALID_SCORING_METRICS, "scoring.metric")
    _check_enum(raw["predictor"]["model_type"], _VALID_PREDICTOR_TYPES, "predictor.model_type")

    # Split fractions must sum to ~1.0
    sp = raw["predictor"]["split"]
    total = sp["train"] + sp["val"] + sp["test"]
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"predictor.split fractions sum to {total}, expected 1.0")

    # KenLM baseline must be explicitly present (even if disabled)
    if "baseline_kenlm" not in raw["lm"]:
        raise KeyError(
            "Config [lm] missing 'baseline_kenlm' section. "
            "Add it with enabled: false if you don't need it."
        )


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _resolve_paths(paths: PathsConfig, root: str) -> PathsConfig:
    """Make all paths absolute relative to the project root."""
    for attr in vars(paths):
        val = getattr(paths, attr)
        if not os.path.isabs(val):
            setattr(paths, attr, os.path.normpath(os.path.join(root, val)))
    return paths


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(config_path: str, project_root: Optional[str] = None) -> Config:
    """
    Load, validate, and return a typed Config from a YAML file.

    Parameters
    ----------
    config_path : str
        Path to the YAML config file.
    project_root : str, optional
        Root directory for resolving relative paths.
        Defaults to two levels above this file (the repo root).

    Returns
    -------
    Config

    Raises
    ------
    FileNotFoundError, KeyError, ValueError
        On missing file, missing keys, or invalid values.
    """
    config_path = os.path.abspath(config_path)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if raw is None:
        raise ValueError(f"Config file is empty: {config_path}")

    _validate(raw)

    if project_root is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    paths = _resolve_paths(PathsConfig(**raw["paths"]), project_root)

    canary = CanaryConfig(**raw["corpus"]["canary"])
    corpus_d = {k: v for k, v in raw["corpus"].items() if k != "canary"}
    corpus = CorpusConfig(**corpus_d, canary=canary)

    # Resolve corpus-level paths relative to project root
    if corpus.base_source and not os.path.isabs(corpus.base_source):
        corpus.base_source = os.path.normpath(os.path.join(project_root, corpus.base_source))
    if corpus.canary.file and not os.path.isabs(corpus.canary.file):
        corpus.canary.file = os.path.normpath(os.path.join(project_root, corpus.canary.file))
    if corpus.candidates_file and not os.path.isabs(corpus.candidates_file):
        corpus.candidates_file = os.path.normpath(os.path.join(project_root, corpus.candidates_file))

    lm_training = LMTrainingConfig(**raw["lm"]["training"])
    baseline_raw = raw["lm"].get("baseline_kenlm", {"enabled": False, "order": 5, "smoothing": "kneser_ney"})
    baseline_kenlm = BaselineKenLMConfig(**baseline_raw)
    lm_d = {k: v for k, v in raw["lm"].items() if k not in ("training", "baseline_kenlm")}
    lm = LMConfig(**lm_d, training=lm_training, baseline_kenlm=baseline_kenlm)

    split = SplitConfig(**raw["predictor"]["split"])
    pred_d = {k: v for k, v in raw["predictor"].items() if k != "split"}
    predictor = PredictorConfig(**pred_d, split=split)

    return Config(
        run_id=str(raw["run_id"]),
        seed=int(raw["seed"]),
        paths=paths,
        corpus=corpus,
        tokenizer=TokenizerConfig(**raw["tokenizer"]),
        lm=lm,
        scoring=ScoringConfig(**raw["scoring"]),
        predictor=predictor,
        evaluation=EvaluationConfig(**raw["evaluation"]),
    )
