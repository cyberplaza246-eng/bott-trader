"""Count live stop-handling failures in errors.log / trading_bot.log."""
import json
import os
import re
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PATTERNS = {
    "stop crossed, emergency stop placed": r"SL already crossed by market",
    "no working stop, re-placing": r"no working SL - re-placing",
    "stop repair rejected": r"SL repair rejected",
    "stop-limit failed, retried as stop-market": r"SL STOP_LIMIT failed",
    "working stop cancelled to upgrade to stop-limit": r"Replacing Lucid STOP_MARKET SL",
    "protective leg not found after submit": r"protective leg .* not found after submit",
    "broker size above tracked, adopting": r"Untracked MNQ size",
    "profit-lock stop move failed": r"PROFIT-LOCK MNQ .* stop move to .* failed",
    "SL move skipped (no entry price)": r"SL move skipped .* no entry price",
}

counts = Counter()
crossed = []
for name in ("errors.log", "trading_bot.log"):
    with open(os.path.join(ROOT, "logs", name), encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                m = json.loads(line).get("message", "")
            except Exception:
                m = line
            for k, p in PATTERNS.items():
                if re.search(p, m):
                    counts[(name, k)] += 1
            if name == "errors.log" and "SL already crossed by market" in m:
                crossed.append(m[:160])

for (name, k), v in sorted(counts.items()):
    print(f"{name:16s} {v:6d}  {k}")
print("\nsample crossed-stop messages:")
for m in crossed[-6:]:
    print("  ", m)
