"""Research 5m ADX ensemble (trend / MR / breakout) for optional LIVE revamp exits.

Isolated from src.strategies / src.strategy so normal live bots stay untouched.
Enable via REVAMP_EXITS=1 (see revamp_exits.py and start_mnq_live_revamp.bat).
"""
from src.research_ensemble.ensemble import EnsembleStrategy
from src.research_ensemble import revamp_exits

__all__ = ["EnsembleStrategy", "revamp_exits"]
