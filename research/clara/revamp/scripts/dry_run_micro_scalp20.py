#!/usr/bin/env python
"""Prove min-hold + $20 TP logic without Lucid."""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.strategy.micro_scalp_20 import (
    MIN_HOLD_SEC,
    OpenPos,
    ScalpState,
    TP_USD,
    check_entry,
    check_exit,
    on_closed,
)


def main() -> int:
    st = ScalpState()
    # Build upward impulse on synthetic ticks
    t0 = 1_000_000.0
    px = 20000.0
    for i in range(30):
        px += 0.4  # ~12 pts over ~8s if 0.4/tick and push every 0.3s
        st.buf.push(px, ts=t0 + i * 0.3)
    sig = check_entry(st, now=t0 + 30 * 0.3)
    assert sig is not None and sig["direction"] == "long", f"expected long entry, got {sig}"
    fill = sig["entry"]
    st.open = OpenPos(
        direction="long",
        entry=fill,
        sl=sig["sl"],
        tp_far=sig["tp_far"],
        entry_ts=t0 + 30 * 0.3,
    )
    entry_ts = st.open.entry_ts
    # At +5s and +$25: must NOT TP yet
    early = check_exit(st.open, fill + 15.0, now=entry_ts + 5.0)
    assert early is None, f"TP must not fire before {MIN_HOLD_SEC}s, got {early}"
    # At +11s and +$25: TP
    late = check_exit(st.open, fill + 15.0, now=entry_ts + 11.0)
    assert late is not None and late[0] == "TP", f"expected TP after min hold, got {late}"
    # Stop always works before min hold
    st.open = OpenPos("long", fill, fill - 10.0, fill + 500, entry_ts)
    sl_hit = check_exit(st.open, fill - 10.0, now=entry_ts + 2.0)
    assert sl_hit is not None and sl_hit[0] == "SL", f"expected SL, got {sl_hit}"
    on_closed(st, "SL", now=entry_ts + 2.0)

    out = {
        "ok": True,
        "min_hold_sec": MIN_HOLD_SEC,
        "tp_usd": TP_USD,
        "early_tp_blocked": True,
        "tp_after_min_hold": True,
        "sl_before_min_hold": True,
    }
    path = os.path.join(ROOT, "data", "micro_scalp20_dry_run.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("DRY RUN PASS", json.dumps(out))
    print("Saved", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
