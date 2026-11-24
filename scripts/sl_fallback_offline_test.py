"""Offline unit test for RITHMIC_SL_ENTRY_FALLBACK (Patch A + B). Mocks only, no broker connection."""
import asyncio, os, sys, threading, unittest
from types import SimpleNamespace
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.broker.rithmic_connector import RithmicConnector  # noqa: E402

OID, SYM = "bot_test1", "MNQ"
ENTRY, SL0, MARKET = 29787.50, 29777.50, 29914.25


def make(record, market=MARKET):
    c = object.__new__(RithmicConnector)          # skip __init__ (no connection)
    c._connected = True
    c._order_lock = threading.RLock()
    c._orders = {OID: dict(record)} if record is not None else {}
    c._run_sync = lambda coro, timeout=15: asyncio.run(coro)
    c._live_market_price = lambda symbol: market
    c.placed = []
    async def fresh(order_id, symbol, side, qty, stop, mkt):
        c.placed.append(stop); return "newsl1"
    async def working(order_id, exclude=""): return ["oldsl1"]
    c.cancelled = []
    async def cancel(sl_id, **kw): c.cancelled.append(sl_id); return True
    c._async_place_fresh_sl = fresh
    c._async_working_sl_ids = working
    c._async_cancel_protective_leg = cancel
    return c


FULL = {"entry_price": ENTRY, "stop_loss": SL0, "symbol": SYM}


class T(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(os.environ, {}, clear=False)
        self.env.start()
        os.environ.pop("RITHMIC_SL_ENTRY_FALLBACK", None)
        self.logs = []
        self.p = mock.patch("src.broker.rithmic_connector.bot_logger")
        self.log = self.p.start()
    def tearDown(self):
        self.p.stop(); self.env.stop()
    def msgs(self):
        return " | ".join(str(c.args[0]) for c in self.log.warning.call_args_list + self.log.info.call_args_list)
    def upd(self, c, new_sl, **kw):
        return c.update_stop_loss(OID, SYM, "BUY", 1, new_sl, 29979.25, **kw)

    # switch OFF
    def test_off_missing_record_same_skip(self):
        c = make(None)
        self.assertFalse(self.upd(c, 29812.50, entry_price=ENTRY, current_sl=SL0))   # args ignored when off
        self.assertEqual(c.placed, [])
        self.assertIn("no entry price", self.msgs())
    def test_off_record_present_works(self):
        c = make(FULL)
        self.assertTrue(self.upd(c, 29812.50)); self.assertEqual(c.placed, [29812.50])

    # switch ON
    def test_on_missing_record_proceeds(self):
        os.environ["RITHMIC_SL_ENTRY_FALLBACK"] = "1"
        c = make(None)
        self.assertTrue(self.upd(c, 29812.50, entry_price=ENTRY, current_sl=SL0))
        self.assertEqual(c.placed, [29812.50]); self.assertEqual(c.cancelled, ["oldsl1"])
        self.assertIn("using fallback entry 29787.50", self.msgs())
        self.assertNotIn("no entry price - keeping", self.msgs())
    def test_on_record_present_unchanged(self):
        os.environ["RITHMIC_SL_ENTRY_FALLBACK"] = "1"
        c = make(FULL)
        # caller hints deliberately different: record must win
        self.assertTrue(self.upd(c, 29812.50, entry_price=1.0, current_sl=99999.0))
        self.assertEqual(c.placed, [29812.50])
        self.assertNotIn("using fallback", self.msgs())
    def test_on_no_hints_still_no_entry_price(self):
        os.environ["RITHMIC_SL_ENTRY_FALLBACK"] = "1"
        c = make(None)
        self.assertFalse(self.upd(c, 29812.50))
        self.assertEqual(c.placed, []); self.assertIn("no entry price", self.msgs())

    # guards still enforced with switch ON + missing record
    def test_on_loosening_rejected(self):
        os.environ["RITHMIC_SL_ENTRY_FALLBACK"] = "1"
        c = make(None)
        self.assertFalse(self.upd(c, 29770.00, entry_price=ENTRY, current_sl=SL0))   # lower than 29777.5 on a long
        self.assertEqual(c.placed, [])
    def test_on_not_tighter_than_hint_rejected(self):
        os.environ["RITHMIC_SL_ENTRY_FALLBACK"] = "1"
        c = make(None)
        self.assertFalse(self.upd(c, 29777.50, entry_price=ENTRY, current_sl=29850.0))
        self.assertEqual(c.placed, [])
    def test_on_wrong_side_of_entry_rejected(self):
        os.environ["RITHMIC_SL_ENTRY_FALLBACK"] = "1"
        c = make(None)   # stop above 29787.5 is profit side; stop below entry (loss side) must be rejected
        self.assertFalse(self.upd(c, 29780.00, entry_price=ENTRY, current_sl=29770.0))
        self.assertEqual(c.placed, [])
    def test_on_not_in_profit_rejected(self):
        os.environ["RITHMIC_SL_ENTRY_FALLBACK"] = "1"
        c = make(None, market=29780.0)   # market below entry
        self.assertFalse(self.upd(c, 29812.50, entry_price=ENTRY, current_sl=SL0))
        self.assertEqual(c.placed, [])
        self.assertIn("trade not in profit", self.msgs())
    def test_on_stop_beyond_market_rejected(self):
        os.environ["RITHMIC_SL_ENTRY_FALLBACK"] = "1"
        c = make(None)
        self.assertFalse(self.upd(c, 29920.00, entry_price=ENTRY, current_sl=SL0))  # above live 29914.25
        self.assertEqual(c.placed, [])

    # Patch B: bot side passes kwargs only when switch on
    def _bot_call(self):
        # Load ONLY _apply_stop_update from the real file via ast (importing the whole bot
        # module would pull in config/ML/logging side effects; keep this test hermetic).
        import ast
        src = open(os.path.join(ROOT, "start_live_rithmic.py"), encoding="utf-8").read()
        fn = next(n for n in ast.walk(ast.parse(src))
                  if isinstance(n, ast.FunctionDef) and n.name == "_apply_stop_update")
        ns = {"os": os, "F3EV": mock.MagicMock(), "Position": object}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "start_live_rithmic.py", "exec"), ns)
        calls = []
        class B:
            def update_stop_loss(self, **kw): calls.append(kw); return True
        pos = SimpleNamespace(direction="long", order_id=OID, symbol=SYM, size=1, tp=29979.25,
                              entry_price=ENTRY, sl=SL0)
        bot = SimpleNamespace(paper_mode=False, paper_orders=False, broker=B(),
                              _is_tighter_sl=lambda p, n, c: True)
        ok = ns["_apply_stop_update"](bot, pos, 29812.50)
        return ok, calls[0]
    def test_bot_off_no_extra_kwargs(self):
        ok, kw = self._bot_call()
        self.assertTrue(ok); self.assertNotIn("entry_price", kw); self.assertNotIn("current_sl", kw)
    def test_bot_on_passes_pos_values(self):
        os.environ["RITHMIC_SL_ENTRY_FALLBACK"] = "1"
        ok, kw = self._bot_call()
        self.assertTrue(ok); self.assertEqual(kw["entry_price"], ENTRY); self.assertEqual(kw["current_sl"], SL0)


if __name__ == "__main__":
    unittest.main(verbosity=2)


