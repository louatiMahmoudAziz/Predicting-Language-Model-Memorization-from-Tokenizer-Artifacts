"""
extract_validate.py
-------------------
Validates memorization risk predictions via extraction experiments.

Responsibilities:
- Attempting to extract candidate strings from a target model via prompting
- Comparing extraction success against predicted risk scores
- Computing extraction-based validation metrics
- Saving validation results to results/metrics/
"""


def validate_via_extraction(candidates_path: str, predictions_path: str, output_dir: str) -> None:
    """Run extraction experiments and validate predicted memorization risk."""
    raise NotImplementedError
