"""Split live P&L into trades the strategy opened vs positions the bot 'adopted' from the broker."""
import json
import os
import re
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

rows = json.load(open(os.path.join(ROOT, "data", "session_pnl.json"), encoding="utf-8"))["rows"]
groups = {"adopted": [], "bot": []}
for r in rows:
    groups["adopted" if r["order_id"].startswith("bot_adopt") else "bot"].append(r["pnl"])
print("session ledger (Oct 28 - Nov 5):")
for k, v in groups.items():
    print(f"  {k:8s} n={len(v):3d} net=${sum(v):,.2f} wins={sum(1 for x in v if x > 0)}")

# trades.log: count EXIT lines; the adopted ones are logged in trading_bot.log by order id
exits = []
with open(os.path.join(ROOT, "logs", "trades.log"), encoding="utf-8", errors="replace") as f:
    for line in f:
        m = json.loads(line).get("message", "")
        x = re.search(r"EXIT (\w+) (long|short) MNQ @ [\d.]+ PnL=\$(-?[\d.]+)", m)
        if x:
            exits.append((x.group(1), float(x.group(3))))
print(f"\ntrades.log MNQ exits (all time): n={len(exits)} net=${sum(p for _, p in exits):,.2f}")
print("  by exit type:", {k: (c, round(sum(p for t, p in exits if t == k), 2)) for k, c in Counter(t for t, _ in exits).items()})

adopt_msgs = Counter()
with open(os.path.join(ROOT, "logs", "trading_bot.log"), encoding="utf-8", errors="replace") as f:
    for line in f:
        if "adopt" not in line.lower():
            continue
        try:
            m = json.loads(line).get("message", "")
        except Exception:
            continue
        adopt_msgs[re.sub(r"bot_adopt_\w+|[\d.]+", "#", m)[:110]] += 1
print("\nmost common adopt messages in trading_bot.log:")
for k, c in adopt_msgs.most_common(8):
    print(f"  {c:5d}  {k}")
