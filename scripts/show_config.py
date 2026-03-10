"""
show_config.py
--------------
CLI utility: load and print a config file for quick inspection.

Usage:
    python scripts/show_config.py configs/colab_mini.yaml
"""

import sys
import os

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import load_config


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/show_config.py <path/to/config.yaml>")
        sys.exit(1)

    cfg = load_config(sys.argv[1])
    print(cfg)


if __name__ == "__main__":
    main()
