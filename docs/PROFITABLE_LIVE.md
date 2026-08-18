# Profitable Live Config (real 1m)

**30-second hybrid is retired.** Real Databento 30s lost. Tight 1m SL/TP grids also lost.

## Locked recipe (v3)

1. **Completed** 15-minute EMA 8 vs 21 (last finished 15m bar only — no unfinished bucket).
2. **Yesterday’s** daily close vs EMA20 must agree with that 15m side (no same-day close leak).
3. Up to **six** RTH slots: **9:35, 10:15, 11:00, 12:00, 13:30, 14:30 ET**. **Every entry** (first lot included) needs daily + 15m EMA8/21 + 60m EMA8/21 in the same direction. **From 12:00 ET on**, also need 15m EMA separation / ATR ≥ 0.45. A **second overlapping lot** is allowed only when that sep ≥ 0.45 (60m already required). Flatten **15:50 ET**.
4. **Overnight is off for live.** Same rules on Globex (18:00–9:20) lost in-sample (PF 0.97). Blind extra windows and SL-reentry spam also fail OOS. **Paper overnight research** (`--paper --overnight-research`) is a separate Globex overlay for zones and journaled paper fills. It does not enable overnight on `--live` and must not set `OVERNIGHT_TRADING=true` on this recipe.
5. Stop = `clip(completed 15m ATR14 × 2.0, 20, 60)` points. 500-pt TP is a cap only.
6. **Do not trail.** Breakeven/trail exits killed the edge on this data. Do not use a short time-stop — cutting winners at 45–90 minutes inflates fill count but collapses IS dollars.

**You do not get ~6 good trades per day on this edge.** Hold-to-EOD plus a 2-lot cap means most days fill 1–3. Average is **~1.8 fills per RTH day**. Six fills happened on 4 of 161 RTH days. The six windows are slots, not a promise.

## Honest real-data result (`data/MNQ_1m.csv`)

Completed bars only (tradeable). Fees in.

| Rule | IS PF / $ | OOS PF / $ | Max DD | t/day |
|------|-----------|------------|--------|-------|
| Old leaky 15m SL30 (same-bar 15m + same-day daily) | 1.30 / $1,617 | 2.14 / $3,413 | $2,007 | — |
| Honest 15m SL30 (baseline) | 1.05 / $311 | 1.41 / $1,293 | $3,044 | — |
| Official 3-window + 2-lot | 1.57 / $6,207 | 1.53 / $3,614 | $1,786 | 1.43 |
| **Official (6 windows, noon+ quality, 2-lot)** | **1.41 / $5,643** | **1.59 / $5,371** | **$2,084** | **1.79** |

OOS months (locked 6-window): Jun +$2,006, Jul +$2,439, Aug +$926. Full sample: 288 trades, WR 33.0%, PF 1.48, **+$11,014**.

### What we tried and killed

- Tight scalps, ORB, VWAP reclaim, open-drive, 10:00 delay: fail or thin.
- **Trailing / move-to-BE:** IS and OOS both lose. Winners need room until 15:50.
- Same-day daily close filter: huge PF, **not tradable** (look-ahead). We do not use it.
- Overnight Globex windows: IS PF 0.97.
- Blind 5 RTH windows (10:30 / 11:30 / 14:30 set): OOS PF 1.08–1.13.
- Re-enter after SL with 20/30/45 min cooldown, cap 6: OOS PF ≤ 1.08 or negative.
- 45–90 min max hold to free later windows: more fills (~3/day) but IS $ collapses.

MES 1m is only a short `.bak` file — not a second independent setup.

## How to run

```bat
.\start_mnq_live.bat
.\start_mnq_forever.bat
.\start_mnq_backtest.bat
```

Research: `python scripts/upgrade_ema15_eod.py` and `python scripts/find_six_trades_ema15.py`

Overnight **paper** research (not live): `.\start_mnq_overnight_paper.bat` or PowerShell `$env:PAPER_USE_RITHMIC="true"; python start_live_mtf_scalping.py --paper --overnight-research --symbols MNQ`. Close Lucid / R|Trader desktop first so the ticker session is free. Live stays RTH.

| Param | Value |
|-------|--------|
| `STRATEGY_MODE` | `ema15_eod` |
| `EMA15_REQUIRE_DAILY` | `true` |
| `EMA15_REQUIRE_60M` | `true` (first lot and all entries) |
| `EMA15_SL_ATR_MULT` / min / max | `2.0` / `20` / `60` |
| `USE_30S_BARS` | `false` |
| `MAX_TRADES_PER_DAY` | `6` (windows 9:35 / 10:15 / 11:00 / 12:00 / 13:30 / 14:30; daily+15m+60m; noon+ still needs sep) |
| Flatten | 15:50 ET |

Live needs ~20 sessions of 1m history so daily EMA20 is real. One contract (two only on high-confidence). Losing months still happen.

## Overnight paper research (not live)

RTH is closed overnight, so this is the mode that can actually run after 18:00 ET. It is **research only**. `--live` / `start_mnq_live.bat` stay RTH, flatten 15:50, and never turn on `OVERNIGHT_TRADING`. Same 15m windows on Globex already failed IS (PF 0.97).

Paper cues (1 MNQ, flatten 09:25 ET so nothing leaks into the locked RTH book):

- **Fade first touch** of a *settled* overnight high or low (wick tags the level, close back inside) with an ATR stop.
- **Break-and-hold** if a completed 1m close holds beyond that settled high/low.

Zones journaled (not all traded): overnight high/low, prior RTH high/low, nearest 50-pt round number, distance to completed 15m EMA21. Fills are tagged `session=overnight_research` in `data/paper_trade_journal.jsonl`. Gemini stores a short note on each close and, after ~20 overnight closes, writes `data/overnight_suggestions.json` — never auto-applied, never writes `mnq_profit_config`.

Close Lucid / R|Trader desktop first so Rithmic can keep the ticker session. Quotes come from the Rithmic ticker (`PAPER_USE_RITHMIC=true`); no orders are sent. Databento 1m is the fallback if ticker login fails.

Desk: http://127.0.0.1:5055 — status **PAPER OVERNIGHT RESEARCH**. Restart the desk if it was already open.

**Local only.** Leave `start_mnq_desk.bat` and `start_mnq_overnight_paper.bat` open (black windows). Closing Cursor is fine if those stay up. Vercel cannot replace them — Flask, Rithmic quotes, and `data/paper_trade_journal.jsonl` are on this machine.

## Trade journal and suggestions (advisory)

The live/paper bot appends every ema15 fill to JSONL and writes a **deterministic WHY** on each close (no Gemini required). After enough history it clusters winners/losers and writes **suggestions you can apply by hand**. It will **not** change `ENTRY_WINDOWS`, stops, `MAX_TRADES_PER_DAY`, or `data/mnq_profit_config.json`.

| What | Where |
|------|--------|
| Live journal | `data/trade_journal.jsonl` |
| Paper journal | `data/paper_trade_journal.jsonl` (overnight paper rows tagged `session=overnight_research`) |
| Overnight zones | `data/overnight_zones.json` |
| Overnight suggestions | `data/overnight_suggestions.json` (after ~20 paper closes; never auto-applied) |
| Suggestions | `data/trade_suggestions.json` |
| Desk UI | http://127.0.0.1:5055 — closed trades, live + backtest expectancy, suggestions (`.\start_mnq_desk.bat`) |
| Window-edge snapshot | `data/window_edge_analysis.json` |

Each close stores: ET timestamps, window, entry/exit, pts, $, hold minutes, exit reason (`Stop` / `EOD flatten` / `TP cap`), market snapshot at entry and exit (daily permission, 15m/60m EMAs, ATR stop, alignment flags), MAE/MFE from 1m bars, R-multiple, and `why.primary_reason` plus supporting `facts[]`.

The desk also has a **closed-trade table** (every fill: times, direction, prices, stop distance, MAE/MFE, exit, 15m/60m/daily, ATR) and **expectancy panels** grouped by window (9:35 / 11:00 / 13:30 plus any extra slots present), longs vs shorts, and trend aligned vs not. Official 1m backtest aggregates live in `data/window_edge_analysis.json` (`python scripts/analyze_window_edge.py`) so you can see backtest truth while the live journal is still small. That snapshot is advisory only — it does **not** write `mnq_profit_config` or change windows/stops/`MAX_TRADES`.

Suggestion gate (env): `SUGGEST_MIN_TRADES` (default **30** closes) and `SUGGEST_MIN_LOSERS` (default **10**). Until then the desk shows progress only. Gemini (`LLM_MODE=advisory`) may add a one-line comment; it cannot change orders. Adaptive / local pattern learners stay **off** for ema15 so they cannot mutate the locked recipe.

