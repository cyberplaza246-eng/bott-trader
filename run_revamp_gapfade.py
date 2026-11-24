"""Wrapper that runs the stock revamp live bot (start_live_rithmic.py, UNMODIFIED) and, only when
GAPFADE_ENABLED=1, adds the NY-open gap-fade sleeve in memory. Same CLI as start_live_rithmic.py.
With GAPFADE_ENABLED unset/0 it is identical to running start_live_rithmic.py."""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import start_live_rithmic as live            # noqa: E402  (import has no side effects beyond the stock module's own)
from src.core import gapfade_sleeve          # noqa: E402

gapfade_sleeve.install(live.LiveRithmicTrader, live)

if __name__ == "__main__":
    live.main()
