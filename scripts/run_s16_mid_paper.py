#!/usr/bin/env python
"""CLI: MNQ S16 mid HybridS15_IB paper bot (Lucid TEST)."""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(ROOT, ".env"))


def main() -> int:
    parser = argparse.ArgumentParser(description="MNQ S16 mid paper (Lucid TEST)")
    parser.add_argument("--duration", type=int, default=0, help="Minutes then exit (0=forever)")
    args = parser.parse_args()
    os.environ["TRADING_MODE"] = "paper"
    os.environ["ALLOW_LIVE"] = "false"
    os.environ["DAY_HYBRID"] = "false"
    os.environ["PAPER_DAY_HYBRID"] = "false"
    os.environ.setdefault("PAPER_USE_RITHMIC", "true")
    os.environ.setdefault("PAPER_RITHMIC_BRACKETS", "true")
    from src.strategy.s16_mid_paper_runner import run_s16_mid_paper

    run_s16_mid_paper(duration_minutes=args.duration)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
