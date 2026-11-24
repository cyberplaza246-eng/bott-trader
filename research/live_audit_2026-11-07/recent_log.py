"""Tally the most frequent message types in the last part of the bot log and the errors log."""
import json
import os
import re
from collections import Counter, deque

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def tail_counts(name, n_lines, top):
    q = deque(maxlen=n_lines)
    with open(os.path.join(ROOT, "logs", name), encoding="utf-8", errors="replace") as f:
        for line in f:
            q.append(line)
    c = Counter()
    for line in q:
        try:
            m = json.loads(line).get("message", "")
        except Exception:
            m = line
        c[re.sub(r"bot_(adopt_)?\w{10}|[\d.]+", "#", m)[:130]] += 1
    print(f"\n== {name}: last {len(q)} lines")
    for k, v in c.most_common(top):
        print(f"{v:6d}  {k}")
    return c


tail_counts("trading_bot.log", 30000, 30)
err = tail_counts("errors.log", 1000000, 20)
unprot = sum(v for k, v in err.items() if k.startswith("UNPROTECTED"))
print(f"\nUNPROTECTED-position messages in errors.log (all time): {unprot}")
