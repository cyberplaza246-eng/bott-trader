"""Feature parity vs the training parquet, and a ban on broker imports."""
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kid_engine as K
import kid_shadow as S


def test_no_broker_imports():
    for name in ("kid_engine.py", "kid_shadow.py"):
        tree = ast.parse((HERE / name).read_text(encoding="utf-8"))
        mods = []
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                mods += [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.module:
                mods.append(n.module)
        bad = [m for m in mods if "rithmic" in m.lower() or "async_rithmic" in m.lower()]
        assert not bad, bad


def test_feature_parity_fixture():
    z = np.load(HERE / "parity_fixture.npz", allow_pickle=False)
    df = pd.DataFrame(dict(datetime=pd.to_datetime(z["datetime_ns"], utc=True), open=z["open"], high=z["high"],
                           low=z["low"], close=z["close"], volume=z["volume"], no_trade=z["no_trade"]))
    cols, norm = K.load_norm()
    _, F, _ = K.build_features(df, cols, norm)
    got = F[1500:1550]
    exp = z["feat"].copy()
    for i, c in enumerate(cols):
        if any(c.startswith(p) for p in K.DROP):
            exp[:, i] = 0.0
    assert got.shape == exp.shape
    err = float(np.max(np.abs(got - exp)))
    assert err == 0.0, err


def test_shadow_module_has_no_order_api():
    text = (HERE / "kid_shadow.py").read_text(encoding="utf-8").lower()
    for word in ("place_order", "rithmic", "lucid", "async_rithmic"):
        assert word not in text, word

def test_plain_speech():
    entry = pd.Timestamp("2026-10-08 17:25:00", tz="UTC")
    exit_ = pd.Timestamp("2026-10-08 17:30:00", tz="UTC")
    assert S.pretty(entry) == "Oct 8, 1:25 PM"
    assert S.entry_line(entry, "LONG", 30932, 30845, 31019) == (
        "Oct 8, 1:25 PM: LONG 1 MNQ at 30932. SL 30845, TP 31019."
    )
    stopped = dict(entry_time=entry, side="LONG", entry=30932, sl=30845, tp=31019,
                   exit_time=exit_, exit=30845, reason="stop", pnl=-176.4)
    assert S.exit_line(stopped) == "Oct 8, 1:30 PM: LONG closed, SL hit at 30845. LOST $176."
    targeted = dict(stopped)
    targeted.update(side="SHORT", reason="target", exit=31222, pnl=147.2)
    assert S.exit_line(targeted) == "Oct 8, 1:30 PM: SHORT closed, TP hit at 31222. MADE $147."
    done = dict(stopped)
    done.update(reason="flat", exit=31391, pnl=35)
    assert S.exit_line(done) == "Oct 8, 1:30 PM: LONG closed, end of session at 31391. MADE $35."
    status_at = pd.Timestamp("2026-10-08 20:20:00", tz="UTC")
    assert S.status_line(status_at, 30996, None) == "Oct 8, 4:20 PM: Price 30996. No trade open."
    op = dict(side="SHORT", entry=30943, sl=31000, tp=30880, since=entry)
    assert S.status_line(status_at, 30923, op) == (
        "Oct 8, 4:20 PM: In a SHORT from 30943 (SL 31000, TP 30880), now up $40."
    )
    assert S.summary_line([stopped]) == "Since Oct 8: 1 pretend trade, down $176."
    now = pd.Timestamp("2026-10-08 20:33:00", tz="UTC")
    assert S.rough_line(0, now) == "Results are rough until about Oct 22, about 10 trading days from now."
    assert "rough" not in S.rough_line(2760, now) or S.rough_line(2760, now).startswith("Enough")
    assert S.px(30932.5) == "30932.5"
    assert S.px(30932.25) == "30932.25"

