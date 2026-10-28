#!/usr/bin/env python
"""CLI: MNQ micro-scalp $20 paper bot (Lucid TEST)."""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(ROOT, ".env"))


def main() -> int:
    parser = argparse.ArgumentParser(description="MNQ micro scalp $20 paper (Lucid)")
    parser.add_argument("--duration", type=int, default=0, help="Minutes then exit (0=forever)")
    args = parser.parse_args()
    from src.strategy.micro_scalp_20_paper_runner import run_micro_scalp20_paper

    run_micro_scalp20_paper(duration_minutes=args.duration)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
