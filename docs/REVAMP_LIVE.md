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
