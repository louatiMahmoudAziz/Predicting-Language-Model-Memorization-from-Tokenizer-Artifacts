"""
run_pipeline.py
---------------
Run the full memorization risk pipeline end-to-end from a single YAML config.

Step order (10 operations)
--------------------------
  1  build_corpus          D_clean.txt, D_canary.txt, manifest.json
  2  train_tokenizer       metadata.json + tokenizer artifacts
  3  extract_features      features/<run_id>.parquet
  4  train_lm  (ref)       models/<run_id>/ref/training_manifest.json
  5  train_lm  (target)    models/<run_id>/target/training_manifest.json
  6  score_bpc (ref)       labels/<run_id>_ref_bpc.parquet
  7  score_bpc (target)    labels/<run_id>_target_bpc.parquet
  8  build_labels          labels/<run_id>_labels.parquet
  9  train_predictor       results/<run_id>/predictor/<run_id>_predictor_manifest.json
 10  eval_metrics          results/<run_id>/eval/<run_id>_eval_manifest.json

Idempotence
-----------
  Each step checks if its key output artifact exists. If it does, the step
  is skipped. Use --force to override all idempotence checks.

Prerequisites
-------------
  Before running, each step validates that its required input files exist.
  Missing prerequisites cause a loud failure with an actionable error.

CLI
---
  python -m src.run_pipeline --config configs/colab_nuke.yaml
  python -m src.run_pipeline --config configs/colab_mini.yaml --force
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path resolution — single source of truth for all artifact locations
# ---------------------------------------------------------------------------

def _resolve_paths(cfg: Any) -> Dict[str, str]:
    """
    Derive every artifact path the pipeline needs from the loaded Config.

    Returns a flat dict keyed by logical name.  Every path is absolute.
    """
    run_id = cfg.run_id

    corpus_dir   = os.path.join(cfg.paths.corpus_dir, run_id)
    tok_dir      = cfg.paths.tokenizer_dir
    models_root  = os.path.dirname(cfg.paths.ref_model_dir)
    labels_dir   = cfg.paths.labels_dir
    features_dir = cfg.paths.features_dir
    results_dir  = cfg.paths.results_dir

    return {
        # Step 1 — build_corpus
        "corpus_dir":          corpus_dir,
        "corpus_manifest":     os.path.join(corpus_dir, "manifest.json"),
        "d_clean":             os.path.join(corpus_dir, "D_clean.txt"),
        "d_canary":            os.path.join(corpus_dir, "D_canary.txt"),

        # Step 2 — train_tokenizer
        "tok_dir":             tok_dir,
        "tok_meta":            os.path.join(tok_dir, "metadata.json"),
        "tok_output_dir":      os.path.dirname(tok_dir),
        "tok_id":              os.path.basename(tok_dir),

        # Step 3 — extract_features
        "features_parquet":    os.path.join(features_dir, f"{run_id}.parquet"),
        "candidates":          cfg.corpus.canary.file,

        # Steps 4-5 — train_lm
        "models_root":         models_root,
        "ref_model_dir":       os.path.join(models_root, run_id, "ref"),
        "target_model_dir":    os.path.join(models_root, run_id, "target"),
        "ref_lm_manifest":     os.path.join(models_root, run_id, "ref", "training_manifest.json"),
        "target_lm_manifest":  os.path.join(models_root, run_id, "target", "training_manifest.json"),

        # Steps 6-7 — score_bpc
        "ref_scores":          os.path.join(labels_dir, f"{run_id}_ref_bpc.parquet"),
        "target_scores":       os.path.join(labels_dir, f"{run_id}_target_bpc.parquet"),

        # Step 8 — build_labels
        "labels_parquet":      os.path.join(labels_dir, f"{run_id}_labels.parquet"),

        # Step 9 — train_predictor
        "predictor_dir":       os.path.join(results_dir, run_id, "predictor"),
        "predictor_manifest":  os.path.join(results_dir, run_id, "predictor",
                                            f"{run_id}_predictor_manifest.json"),

        # Step 10 — eval_metrics
        "eval_dir":            os.path.join(results_dir, run_id, "eval"),
        "eval_manifest":       os.path.join(results_dir, run_id, "eval",
                                            f"{run_id}_eval_manifest.json"),
    }


# ---------------------------------------------------------------------------
# Prerequisite checker
# ---------------------------------------------------------------------------

def _require(paths: List[str], step_name: str) -> None:
    """Fail loudly if any required input file is missing."""
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        msg_lines = [f"Step '{step_name}' cannot run — missing prerequisites:"]
        for p in missing:
            msg_lines.append(f"  {p}")
        msg_lines.append("Run earlier pipeline steps first, or check your config.")
        raise FileNotFoundError("\n".join(msg_lines))


# ---------------------------------------------------------------------------
# Individual step runners
# ---------------------------------------------------------------------------

def _step_build_corpus(cfg: Any, P: Dict[str, str]) -> str:
    from src.build_corpus import build_corpus
    _require([cfg.corpus.base_source, cfg.corpus.canary.file], "build_corpus")

    build_corpus(
        base_source=cfg.corpus.base_source,
        canary_file=cfg.corpus.canary.file,
        repetitions=cfg.corpus.canary.repetitions,
        budget_type=cfg.corpus.budget_type,
        budget_value=cfg.corpus.budget_value,
        output_dir=cfg.paths.corpus_dir,
        run_id=cfg.run_id,
        seed=cfg.seed,
    )
    return P["corpus_manifest"]


def _step_train_tokenizer(cfg: Any, P: Dict[str, str]) -> str:
    from src.train_tokenizer import train_tokenizer
    _require([P["d_clean"]], "train_tokenizer")

    meta_path = train_tokenizer(
        corpus_path=P["d_clean"],
        tok_type=cfg.tokenizer.type,
        vocab_size=cfg.tokenizer.vocab_size,
        min_frequency=cfg.tokenizer.min_frequency,
        special_tokens=cfg.tokenizer.special_tokens,
        output_dir=P["tok_output_dir"],
        tok_id=P["tok_id"],
        config_snapshot={
            "run_id":        cfg.run_id,
            "seed":          cfg.seed,
            "tok_type":      cfg.tokenizer.type,
            "vocab_size":    cfg.tokenizer.vocab_size,
            "min_frequency": cfg.tokenizer.min_frequency,
            "special_tokens": cfg.tokenizer.special_tokens,
        },
    )
    return meta_path


def _step_extract_features(cfg: Any, P: Dict[str, str]) -> str:
    from src.extract_features import extract_features
    _require([P["tok_meta"], P["candidates"]], "extract_features")

    out = extract_features(
        target_metadata_path=P["tok_meta"],
        candidates_path=P["candidates"],
        output_dir=cfg.paths.features_dir,
        run_id=cfg.run_id,
    )
    return out


def _step_train_lm(cfg: Any, P: Dict[str, str], role: str) -> str:
    from src.train_lm import train_lm

    corpus = P["d_clean"] if role == "ref" else P["d_canary"]
    _require([corpus, P["tok_meta"]], f"train_lm_{role}")

    t = cfg.lm.training
    manifest = train_lm(
        corpus_path=corpus,
        tokenizer_metadata_path=P["tok_meta"],
        output_dir=P["models_root"],
        run_id=cfg.run_id,
        role=role,
        d_model=cfg.lm.d_model,
        n_heads=cfg.lm.n_heads,
        n_layers=cfg.lm.n_layers,
        d_ff=cfg.lm.d_ff,
        dropout=cfg.lm.dropout,
        max_seq_len=cfg.lm.max_seq_len,
        batch_size=t.batch_size,
        learning_rate=t.learning_rate,
        max_steps=t.max_steps,
        warmup_steps=t.warmup_steps,
        weight_decay=t.weight_decay,
        log_every=t.log_every,
        eval_every=t.eval_every,
        checkpoint_every=t.checkpoint_every,
        seed=cfg.seed,
        require_matched_ref=(role == "target"),
    )
    return manifest


def _step_score_bpc(cfg: Any, P: Dict[str, str], role: str) -> str:
    from src.score_bpc import score_bpc

    model_dir = P["ref_model_dir"] if role == "ref" else P["target_model_dir"]
    _require(
        [os.path.join(model_dir, "training_manifest.json"),
         P["tok_meta"], P["candidates"]],
        f"score_bpc_{role}",
    )

    out = score_bpc(
        model_dir=model_dir,
        tokenizer_metadata_path=P["tok_meta"],
        candidates_path=P["candidates"],
        output_dir=cfg.paths.labels_dir,
        run_id=cfg.run_id,
        role=role,
        batch_size=cfg.scoring.batch_size,
        add_bos=cfg.scoring.add_bos,
        add_eos=cfg.scoring.add_eos,
        allow_truncation=cfg.scoring.allow_truncation,
    )
    return out


def _step_build_labels(cfg: Any, P: Dict[str, str]) -> str:
    from src.build_labels import build_labels
    _require([P["ref_scores"], P["target_scores"]], "build_labels")

    out = build_labels(
        ref_scores_path=P["ref_scores"],
        target_scores_path=P["target_scores"],
        output_dir=cfg.paths.labels_dir,
        run_id=cfg.run_id,
    )
    return out


def _step_train_predictor(cfg: Any, P: Dict[str, str]) -> str:
    from src.train_predictor import train_predictor
    _require([P["features_parquet"], P["labels_parquet"]], "train_predictor")

    out = train_predictor(
        features_path=P["features_parquet"],
        labels_path=P["labels_parquet"],
        output_dir=P["predictor_dir"],
        run_id=cfg.run_id,
        model_type=cfg.predictor.model_type,
        split_train=cfg.predictor.split.train,
        split_val=cfg.predictor.split.val,
        seed=cfg.seed,
        max_iter=cfg.predictor.max_iter,
        n_estimators=cfg.predictor.n_estimators,
        threshold=cfg.predictor.threshold,
        ranking_k=cfg.evaluation.ranking_k,
    )
    return out


def _step_eval_metrics(cfg: Any, P: Dict[str, str]) -> str:
    from src.eval_metrics import eval_metrics
    _require([P["predictor_manifest"]], "eval_metrics")

    out = eval_metrics(
        predictor_dir=P["predictor_dir"],
        run_id=cfg.run_id,
        output_dir=P["eval_dir"],
        fpr_0_1_min_negatives=cfg.evaluation.fpr_0_1_min_negatives,
        ranking_k=cfg.evaluation.ranking_k,
    )
    return out


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

# (step_number, name, artifact_key, runner)
_STEPS = [
    (1,  "build_corpus",     "corpus_manifest",    lambda c, p: _step_build_corpus(c, p)),
    (2,  "train_tokenizer",  "tok_meta",           lambda c, p: _step_train_tokenizer(c, p)),
    (3,  "extract_features", "features_parquet",   lambda c, p: _step_extract_features(c, p)),
    (4,  "train_lm_ref",     "ref_lm_manifest",    lambda c, p: _step_train_lm(c, p, "ref")),
    (5,  "train_lm_target",  "target_lm_manifest", lambda c, p: _step_train_lm(c, p, "target")),
    (6,  "score_bpc_ref",    "ref_scores",         lambda c, p: _step_score_bpc(c, p, "ref")),
    (7,  "score_bpc_target", "target_scores",      lambda c, p: _step_score_bpc(c, p, "target")),
    (8,  "build_labels",     "labels_parquet",     lambda c, p: _step_build_labels(c, p)),
    (9,  "train_predictor",  "predictor_manifest", lambda c, p: _step_train_predictor(c, p)),
    (10, "eval_metrics",     "eval_manifest",      lambda c, p: _step_eval_metrics(c, p)),
]


def run_pipeline(
    config_path: str,
    *,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Run the full memorization risk pipeline.

    Parameters
    ----------
    config_path : path to YAML config
    force       : if True, re-run all steps regardless of existing outputs

    Returns
    -------
    dict  Pipeline summary with status of each step and artifact paths.
    """
    from src.config import load_config

    config_path = os.path.abspath(config_path)
    cfg = load_config(config_path)
    P = _resolve_paths(cfg)

    logger.info("=" * 70)
    logger.info("PIPELINE START  run_id=%s  config=%s", cfg.run_id, config_path)
    logger.info("=" * 70)

    summary: Dict[str, Any] = {
        "run_id":     cfg.run_id,
        "config":     config_path,
        "started_at": datetime.datetime.utcnow().isoformat() + "Z",
        "force":      force,
        "steps":      [],
    }

    pipeline_ok = True

    for step_num, step_name, artifact_key, runner in _STEPS:
        artifact_path = P[artifact_key]
        step_record: Dict[str, Any] = {
            "step":     step_num,
            "name":     step_name,
            "artifact": artifact_path,
        }

        # --- Idempotence check ---
        if not force and os.path.isfile(artifact_path):
            logger.info(
                "SKIP  %02d  %-20s  (exists: %s)", step_num, step_name, artifact_path,
            )
            step_record["status"] = "skipped"
            summary["steps"].append(step_record)
            continue

        # --- Run ---
        logger.info("RUN   %02d  %-20s ...", step_num, step_name)
        t0 = time.time()

        try:
            result_path = runner(cfg, P)
            elapsed = time.time() - t0
            logger.info(
                "DONE  %02d  %-20s  (%.1fs) -> %s",
                step_num, step_name, elapsed, result_path,
            )
            step_record["status"] = "done"
            step_record["elapsed_s"] = round(elapsed, 1)
            step_record["result"] = str(result_path)

        except Exception as exc:
            elapsed = time.time() - t0
            logger.error(
                "FAIL  %02d  %-20s  (%.1fs): %s",
                step_num, step_name, elapsed, exc,
            )
            step_record["status"] = "failed"
            step_record["elapsed_s"] = round(elapsed, 1)
            step_record["error"] = str(exc)
            pipeline_ok = False
            summary["steps"].append(step_record)
            break

        summary["steps"].append(step_record)

    summary["finished_at"] = datetime.datetime.utcnow().isoformat() + "Z"
    summary["success"] = pipeline_ok

    # Write pipeline manifest
    manifest_dir = os.path.join(cfg.paths.results_dir, cfg.run_id)
    os.makedirs(manifest_dir, exist_ok=True)
    manifest_path = os.path.join(manifest_dir, f"{cfg.run_id}_pipeline.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    if pipeline_ok:
        logger.info("=" * 70)
        logger.info("PIPELINE COMPLETE  run_id=%s  manifest=%s", cfg.run_id, manifest_path)
        logger.info("=" * 70)
    else:
        logger.error("=" * 70)
        logger.error("PIPELINE FAILED at step %02d. See log above.", summary["steps"][-1]["step"])
        logger.error("=" * 70)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.run_pipeline",
        description="Run the full memorization risk pipeline from a YAML config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config", required=True, metavar="YAML",
        help="Path to the project YAML config file.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-run all steps, ignoring existing outputs.",
    )
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

    summary = run_pipeline(args.config, force=args.force)

    if not summary["success"]:
        sys.exit(1)

    print(f"\nPipeline complete.  Manifest: "
          f"{os.path.join(summary['steps'][-1].get('artifact', '?'))}")


if __name__ == "__main__":
    main()
