REVAMP LIVE (Rithmic orders) — 2026-10-06
=========================================

Winner exits from overnight research:
  combo_tri__trail_mult4.0__tgt_breakout_r2.0__tgt_mr_bb_far

How to start
------------
  ORIGINAL day bot (ema15_eod, unchanged):
    C:\Users\shawa\Projects\bott-trader\start_mnq_live.bat

  REVAMP live (this path — real Rithmic orders):
    C:\Users\shawa\Projects\bott-trader\start_mnq_live_revamp.bat

  Ensemble live WITHOUT revamp (original 0.6 ATR trail):
    C:\Users\shawa\Projects\bott-trader\START_ENSEMBLE_VISIBLE.bat

  Restore whole tree from backup if needed:
    C:\Users\shawa\Projects\bott-trader-ORIGINAL-2026-10-06

What revamp live does
---------------------
  Launches start_live_rithmic.py with REVAMP_EXITS=1.
  That is the real Rithmic order path (not the paper research engine alone).

  When REVAMP_EXITS=1:
    Entries: research 5m ADX router (trend / mean-reversion / breakout)
    Exits:
      - trend      -> ATR trail x4.0 (no fixed TP; profit locks still apply)
      - breakout   -> fixed 2R take-profit, trailing OFF
      - MR         -> target = far Bollinger band (bb_far), trailing OFF
    Profit locks (unchanged, from .env):
      +$100 uPnL -> SL locks +$15
      +$150 uPnL -> SL locks +$50

  When REVAMP_EXITS is unset/0 (default):
    start_live_rithmic.py keeps the prior AI ensemble + TRAIL_ATR_MULT=0.6.

Flags (env; all default OFF / original)
---------------------------------------
  REVAMP_EXITS=1
  REVAMP_TRAIL_ATR_MULT=4.0
  REVAMP_BREAKOUT_TARGET_R=2.0
  REVAMP_MR_TARGET=bb_far

Code
----
  src/research_ensemble/*          (ported research ensemble + revamp_exits)
  src/research_ensemble/live_signals.py
  start_live_rithmic.py            (hooks; default OFF)
  start_mnq_live_revamp.bat        (turns flags ON and launches live)

Do NOT run start_mnq_live.bat and start_mnq_live_revamp.bat at the same time
(one Lucid / Rithmic login). Close Lucid desktop first.

Overnight / Globex entries (2026-10-06)
--------------------------------------
  start_mnq_live_revamp.bat sets:
    OVERNIGHT_TRADING=true
    SESSION_MODE=extended
    SCALP_SESSIONS=off
    REVAMP_RTH_ONLY=0

  That turns off the research ensemble RTH-only gate (was 09:30-16:00 ET only)
  so the 5m ADX router may enter across CME Globex:
    Sun 6:00 PM – Fri 5:00 PM ET (daily halt ~5:00–6:00 PM ET Mon–Thu).

  Thin UTC bad-hour blacklist (0/4/21) is also cleared while revamp overnight is on.
  No separate daily EOD flatten on this path (ema15_eod 15:50 flatten is the day bot).
  To restore RTH-only: set REVAMP_RTH_ONLY=1 (or OVERNIGHT_TRADING=false + SESSION_MODE=rth).
