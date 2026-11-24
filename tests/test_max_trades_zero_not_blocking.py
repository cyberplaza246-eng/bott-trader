import importlib.util, sys, pathlib, inspect
from unittest.mock import MagicMock
import pytest
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.research_ensemble import live_display as new_ld

def _load_bak():
    p = ROOT / "src/research_ensemble/live_display.py.bak_20261007_maxtrades"
    spec = importlib.util.spec_from_loader("ld_bak", loader=None)
    mod = importlib.util.module_from_spec(spec)
    exec(compile(p.read_text(encoding="utf-8"), str(p), "exec"), mod.__dict__)
    return mod

GATES = dict(daily_loss_limit=500, daily_pnl=0.0, max_trades=0, daily_trades=0,
             open_count=0, max_positions=3, cooldown=0, loss_cooldown_min=0)

def test_old_display_said_stopped_new_does_not():
    old = _load_bak()
    assert "max trades" in (old.gate_block_reason(GATES) or "")
    assert new_ld.gate_block_reason(GATES) is None

def test_display_is_output_only_and_runs_after_entry_decision():
    src = (ROOT / "start_live_rithmic.py").read_text(encoding="utf-8", errors="ignore")
    # live_display's gate functions are never referenced by the bot itself
    assert "gate_block_reason" not in src and "_plain_gate" not in src
    loop = src[src.index("signal = self.check_entry_signal(symbol, df)"):]
    i_append = loop.index("cycle_candidates.append(signal)")
    i_print = loop.index("self._print_revamp_reasoning(")
    assert i_append < i_print  # signal queued before display ever runs
    m = src[src.index("def _print_revamp_reasoning"):]
    m = m[:m.index("\n    def ", 10)]
    assert "return" not in m.replace("-> None", "")  # returns nothing to caller

def test_real_daily_limit_gate_with_zero_cap():
    import datetime as dt
    spec = importlib.util.spec_from_file_location("slr", ROOT / "start_live_rithmic.py")
    src = (ROOT / "start_live_rithmic.py").read_text(encoding="utf-8", errors="ignore")
    fn_src = src[src.index("    def check_daily_limits"):src.index("    def check_entry_signal")]
    import textwrap
    ns = {"datetime": dt.datetime}
    exec(textwrap.dedent(fn_src), ns)
    self = MagicMock(current_date=dt.datetime.now().date(), daily_pnl=0.0,
                     daily_loss_limit=500, max_trades_per_day=0, daily_trades=0)
    assert ns["check_daily_limits"](self) is True
    self.daily_trades = 50
    assert ns["check_daily_limits"](self) is True
