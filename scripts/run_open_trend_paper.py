#!/usr/bin/env python
"""CLI for the MNQ OpenTrend paper bot (Lucid TEST only)."""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(ROOT, ".env"))


def main() -> int:
    parser = argparse.ArgumentParser(description="MNQ OpenTrend paper (Lucid TEST)")
    parser.add_argument("--duration", type=int, default=0, help="Minutes then exit (0=forever)")
    args = parser.parse_args()
    os.environ["TRADING_MODE"] = "paper"
    os.environ["ALLOW_LIVE"] = "false"
    os.environ["PAPER_USE_RITHMIC"] = "true"
    os.environ["PAPER_RITHMIC_BRACKETS"] = "true"
    from src.strategy.open_trend_paper_runner import run_open_trend_paper

    run_open_trend_paper(duration_minutes=args.duration)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
