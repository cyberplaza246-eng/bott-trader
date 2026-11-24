"""
Trade journal for hybrid scalp — append-only JSONL with in-memory ring buffer.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.utils.logger import bot_logger

TRADE_JOURNAL_PATH = "data/trade_journal.jsonl"
MAX_JOURNAL_TRADES = 200


class TradeJournal:
    """Persist closed hybrid trades; keep last N in memory."""

    def __init__(self, path: str = TRADE_JOURNAL_PATH, max_trades: int = MAX_JOURNAL_TRADES):
        self.path = path
        self.max_trades = max_trades
        self.trades: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            loaded: List[Dict[str, Any]] = []
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        loaded.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            self.trades = loaded[-self.max_trades :]
            bot_logger.info(f"Trade journal loaded: {len(self.trades)} trades from {self.path}")
        except Exception as e:
            bot_logger.warning(f"Could not load trade journal: {e}")

    def append(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Append one closed trade; returns normalized record."""
        with self._lock:
            rec = dict(record)
            rec.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str) + "\n")
            self.trades.append(rec)
            if len(self.trades) > self.max_trades:
                self.trades = self.trades[-self.max_trades :]
            return rec

    def recent(self, n: Optional[int] = None) -> List[Dict[str, Any]]:
        if n is None:
            return list(self.trades)
        return list(self.trades[-n:])

    def count(self) -> int:
        return len(self.trades)
