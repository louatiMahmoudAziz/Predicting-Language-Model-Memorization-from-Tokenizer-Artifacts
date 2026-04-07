#!/usr/bin/env python3
"""
run_pretrained.py
-----------------
Convenience script to run the full pretrained evaluation pipeline
from a YAML config file.

Reads the pretrained: section from a config and dispatches to
src.pretrained_eval.run_pretrained_eval().

Usage:
    python scripts/run_pretrained.py --config configs/pythia_1b.yaml
    python scripts/run_pretrained.py --config configs/pythia_1b.yaml --extract
"""

import argparse
import logging
import os
import sys
import yaml

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run pretrained evaluation from YAML config")
    parser.add_argument("--config", required=True, help="YAML config with pretrained: section")
    parser.add_argument("--extract", action="store_true", help="Also run extraction validation")
    parser.add_argument("--device", default="auto", help="Device override")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if "pretrained" not in cfg:
        logger.error("Config %s has no 'pretrained:' section", args.config)
        sys.exit(1)

    pc = cfg["pretrained"]

    # Resolve paths relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    candidates = pc["candidates"]
    if not os.path.isabs(candidates):
        candidates = os.path.join(project_root, candidates)

    output_dir = os.path.join(project_root, "results", pc["run_id"])

    # Run main evaluation
    from src.pretrained_eval import run_pretrained_eval
    summary = run_pretrained_eval(
        target_model_name=pc["target_model"],
        candidates_path=candidates,
        output_dir=output_dir,
        run_id=pc["run_id"],
        ref_model_name=pc.get("ref_model"),
        batch_size=pc.get("batch_size", 32),
        max_length=pc.get("max_length", 1024),
        device=args.device,
        seed=pc.get("seed", 42),
    )

    print(f"\nPretrained eval complete. Results in: {output_dir}")

    # Optional: run extraction validation
    if args.extract:
        from src.extract_validate import validate_via_extraction

        logger.info("Running extraction validation...")
        extract_result = validate_via_extraction(
            candidates_path=candidates,
            output_dir=os.path.join(output_dir, "extraction"),
            run_id=pc["run_id"],
            pretrained_model_name=pc["target_model"],
            predictions_path=os.path.join(output_dir, f"{pc['run_id']}_labels.parquet"),
            max_candidates=500,  # limit for speed
            device=args.device,
        )
        print(f"Extraction validation complete. Metrics: {extract_result['metrics']}")


if __name__ == "__main__":
    main()
