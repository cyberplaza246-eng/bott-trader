"""
Trade-alert tests. Run from the repo root with the venv python.

  python tools\trade_alert_test.py              offline mock suite (no sound, no email, no orders)
  python tools\trade_alert_test.py --sound      play the ENTRY and EXIT beeps on this PC
  python tools\trade_alert_test.py --send-email send ONE real test email (uses your .env / env password)
"""
import logging
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(ROOT, ".env"))
    except Exception:
        pass


if "--sound" in sys.argv:
    from src.utils import trade_alerts as ta
    import winsound
    print("ENTRY beep (3 rising tones)...")
    ta._play("ENTRY"); time.sleep(1.0)
    print("EXIT beep (2 falling tones)...")
    ta._play("EXIT"); time.sleep(1.0)
    print("done. (If you heard nothing: check Windows volume / output device.)")
    sys.exit(0)

if "--send-email" in sys.argv:
    _load_env()
    from src.utils import trade_alerts as ta
    to, user, frm, pw, host, port = ta._email_cfg()
    print(f"to={to} smtp_user={user or '(none)'} host={host}:{port} password_set={bool(pw)}")
    if not (user and pw):
        print("No app password found. Set TRADE_ALERT_SMTP_APP_PASSWORD first.")
        sys.exit(2)
    try:
        ta._send_email_blocking("[bott-trader] TEST alert - ignore", "This is a test of the trade alert email. No trade was placed.")
        print("SENT OK - check", to)
    except Exception as e:
        print("FAILED:", type(e).__name__, str(e)[:200])
        sys.exit(1)
    sys.exit(0)

# ---------------------------------------------------------------- offline mock suite
for k in list(os.environ):
    if k.startswith("TRADE_ALERT") or k in ("SMTP_PASSWORD", "EMAIL_PASS", "EMAIL_PASSWORD", "SMTP_USER", "EMAIL_USER", "EMAIL_FROM"):
        del os.environ[k]
os.environ["TRADE_ALERT_MERGE_SECONDS"] = "0.3"
os.environ["TRADE_ALERT_SMTP_USER"] = "sender@example.test"
os.environ["TRADE_ALERT_SMTP_APP_PASSWORD"] = "mock-not-a-real-password"

from src.utils import trade_alerts as ta  # noqa: E402

beeps, mails = [], []
ta._play = lambda kind: beeps.append(kind)


class FakeSMTP:
    fail = False
    def __init__(self, host, port, timeout=None):
        assert timeout, "SMTP must have a timeout"
        self.host, self.port = host, port
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def starttls(self): pass
    def login(self, u, p): pass
    def send_message(self, msg):
        if FakeSMTP.fail:
            raise OSError("mock network down")
        mails.append((msg["To"], msg["Subject"], msg.get_payload()))


ta.smtplib.SMTP = FakeSMTP

bot = logging.getLogger("TradingBot"); bot.setLevel(logging.INFO)
trd = logging.getLogger("Trades"); trd.setLevel(logging.INFO)
ta.install(bot, trd); ta.install(bot, trd)  # idempotent
assert sum(isinstance(h, ta._TradeAlertHandler) for h in bot.handlers) == 1

fails = []
def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond: fails.append(name)

def reset():
    beeps.clear(); mails.clear(); ta._SENT_TIMES.clear(); ta._WARNED.clear()
    FakeSMTP.fail = False
def settle(): time.sleep(0.9)

# 1 entry: connector line + ENTRY line (real formats from the live log) -> ONE merged alert
reset()
bot.info("Rithmic order placed: bot_ab12cd34ef buy 1 MNQ | brackets=native SL=yes TP=yes")
trd.info("ENTRY long MNQ @ 29732.50 SL=29648.00 TP=29850.75 conf 0.72 size 1")
settle()
check("entry -> 1 beep", beeps == ["ENTRY"])
check("entry -> 1 merged email to paraflix246", len(mails) == 1 and mails[0][0] == "paraflix246@gmail.com")
check("email has price + qty + symbol", mails and "29732.50" in mails[0][1] and "MNQ" in mails[0][1] and " 1 " in mails[0][1])

# 2 ADOPTED position is not an entry
reset()
bot.warning("ADOPTED untracked MNQ BUY x1 @ 29732.50 as bot_adopt_b1e1397331: SL=existing 29648.00 TP=none")
trd.info("EXIT STOP_LOSS long MNQ @ 29663.50 PnL=$-140.50")
settle()
check("adopted + exit(off by default) -> no beep, no email", not beeps and not mails)

# 3 exits optional flag
reset(); os.environ["TRADE_ALERT_EXITS"] = "1"
trd.info("EXIT STOP_LOSS long MNQ @ 29663.50 PnL=$-140.50")
settle()
check("exits flag on -> exit beep + email with PnL", beeps == ["EXIT"] and len(mails) == 1 and "29663.50" in mails[0][1] and "-140.50" in mails[0][2])
os.environ.pop("TRADE_ALERT_EXITS")

# 4 switches off
reset(); os.environ["TRADE_ALERT_SOUND"] = "0"
bot.info("Rithmic order placed: bot_x1 sell 1 MNQ | brackets=native SL=yes TP=yes"); settle()
check("sound=0 -> no beep but email still sent", not beeps and len(mails) == 1)
reset(); os.environ["TRADE_ALERT_SOUND"] = "1"; os.environ["TRADE_ALERT_EMAIL"] = "0"
bot.info("Rithmic order placed: bot_x2 sell 1 MNQ | brackets=native SL=yes TP=yes"); settle()
check("email=0 -> beep but no email", beeps == ["ENTRY"] and not mails)
os.environ.pop("TRADE_ALERT_EMAIL")

# 5 SMTP failure never raises / blocks
reset(); FakeSMTP.fail = True
t0 = time.time()
bot.info("Rithmic order placed: bot_x3 buy 1 MNQ | brackets=native SL=yes TP=yes")
check("logging call returns instantly even with SMTP down", time.time() - t0 < 0.2)
settle(); check("smtp failure swallowed (no mail, no crash)", not mails)

# 6 missing password -> silent skip, one line, no crash
reset(); os.environ.pop("TRADE_ALERT_SMTP_APP_PASSWORD")
bot.info("Rithmic order placed: bot_x4 buy 1 MNQ | brackets=native SL=yes TP=yes"); settle()
check("no password -> no email, sound still plays", not mails and beeps == ["ENTRY"])
os.environ["TRADE_ALERT_SMTP_APP_PASSWORD"] = "mock-not-a-real-password"

# 7 kid / ExtremeSL excluded
reset(); old = sys.argv[0]; sys.argv[0] = r"C:\x\tools\neat_live_kid.py"
bot.info("Rithmic order placed: bot_x5 buy 1 MNQ | brackets=native SL=yes TP=yes"); settle()
check("kid script -> excluded", not beeps and not mails)
sys.argv[0] = old

# 8 flood cap
reset(); os.environ["TRADE_ALERT_MAX_PER_HOUR"] = "2"
for i in range(4):
    bot.info(f"Rithmic order placed: bot_c{i} buy 1 MNQ | brackets=native SL=yes TP=yes"); time.sleep(0.5)
settle(); check("flood cap limits emails to 2", len(mails) == 2)
os.environ.pop("TRADE_ALERT_MAX_PER_HOUR")

# 9 handler garbage tolerance
reset()
try:
    bot.info(None); bot.info("%s", object()); trd.info("ENTRY | Pair: MNQ | Type: BUY | Price: 1.0 | SL: 0 | TP: 0 | Lot Size: 1"); settle()
    ok = True
except Exception:
    ok = False
check("odd log input never raises", ok)

print("\nALL PASSED" if not fails else f"\nFAILED: {fails}")
sys.exit(1 if fails else 0)