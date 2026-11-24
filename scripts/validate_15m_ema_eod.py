#!/usr/bin/env python3
"""Run the official locked-recipe backtest."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.backtest_ema15_official import main

if __name__ == "__main__":
    raise SystemExit(main())
