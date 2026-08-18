# Profitable Live Config (MNQ Hybrid)

**Verdict:** Live recipe = **hybrid ultra_fast** (continuation-heavy + momentum burst), **RTH liquid windows only**.

## Why this mode

| Evidence | Result |
|----------|--------|
| Hybrid ultra_fast backtest (MNQ Dec–Mar, fees in) | 1105 trades, WR 57.4%, **PF 2.42** (RSI gate ON) |
| Ultra_fast no RSI gate | 1107 trades, WR 58.8%, **PF 2.61** |
| Strict hybrid baseline | 945 trades, WR 44.8%, PF 1.11 |
| Momentum Version B (older) | PF ~1.68 |
| Strict pullback Version A | **PF 0.73 — losing** (do not use strict A) |
| FVS-1 | 1 trade in sample — not ready for live entries |
| Live journal (14 trades, all overnight hours UTC 1–6) | WR 28.6%, **−$74** — Globex chop killed edge |
| Last ~10 live sessions | **−$113** mostly overnight / extended |

Production therefore locks **ultra_fast hybrid params**, turns **overnight off**, and filters **SCALP_SESSIONS** to liquid RTH (morning + afternoon; skip lunch chop).

## Locked live recipe

Start with:

```bat
.\start_mnq_live.bat
```

Leave it running unattended (restarts on crash, no YES prompt):

```bat
.\start_mnq_forever.bat
```

Optional: `.\add_to_startup.bat` so Windows launches `start_mnq_forever.bat` at login.

If `.env` does not set `SCALP_MODE` / `STRATEGY_MODE`, the bot now **locks hybrid ultra_fast automatically** from `data/mnq_profit_config.json` (`scalp_hybrid.profit_mode`). It no longer falls through to overnight `full_adaptive`. Open positions are flattened at RTH close, lunch/window close, daily loss limit, and Ctrl+C.

Key `.env` (also mirrored in `data/mnq_profit_config.json` → `scalp_hybrid`):

| Param | Value | Role |
|-------|-------|------|
| `SCALP_MODE` | `hybrid` | Pullback A + Continuation B + burst |
| `SCALP_AGGRESSIVE` / `SCALP_FAST_MODE` | `true` | Ultra-fast frequency path |
| `SCALP_SL_PTS` / `SCALP_TP_PTS` | `6` / `14` | Winner brackets |
| `MAX_HOLD_SECONDS` | `30` | TIME exit = rotation edge |
| `SCALP_CONTINUATION_ADX_MIN` | `18` | Continuation-friendly |
| `SCALP_ADX_MIN` | `15` | Aggressive floor |
| `SCALP_SETUP_BARS` / window | `2` / `60s` | Faster arming |
| `SCALP_TREND_MODE` | `vwap` | Avoid dir=0 chop |
| `SCALP_BREAKEVEN_*` + trail | BE @ 35% to TP, trail on | Protect winners |
| `MNQ_MAX_POSITIONS` | `2` | Cap exposure |
| `LOSS_COOLDOWN_MINUTES` | `15` | After any loss |
| `MTF_MAX_CONSEC_LOSSES` | `3` → 30m pause | Halt spirals |
| `OVERNIGHT_TRADING` | `false` | RTH outer session |
| `SESSION_MODE` | `rth` | Mon–Fri 9:30–4:30 ET |
| `SCALP_SESSIONS` | `morning=09:30-12:00;afternoon=13:30-16:00` | Skip Globex + lunch |
| `USE_DEEPSEEK_LEARNER` | optional | Fail-open; local patterns can block |
| FVS-1 | log-only / off | No live FVS entries without evidence |

Startup prints:

```text
PROFIT MODE: hybrid ultra_fast (continuation-heavy + burst)
```

## Safety

- Max 2 MNQ positions, daily trade cap 20, daily half-stop on
- ATR-bounded SL/TP + min R:R
- Breakeven + trail, max hold 30s
- Loss cooldown + consecutive-loss pause
- Ghost-flat grace / broker position block remain on
- Learner never required for entries (fail-open if API down)

## Caveats (honest)

1. **Simulator ≠ live** — Lucid TEST010 / `RITHMIC_TRADE_ROUTE=simulator` fills and latency differ from production.
2. **Backtest ≠ live** — 30s bars in backtest are synthetic (2× from 1m); real SECOND_BAR timing and flow differ.
3. **Ultra-fast exits on TIME** — most wins are small TIME exits, not TP; expect frequent rotation, not home runs.
4. **Do not re-enable overnight** until RTH PF is proven live over multiple sessions.
5. **Do not enable FVS-1 live entries** until a multi-trade backtest beats hybrid.

## Restart checklist

1. Confirm `.env`: `OVERNIGHT_TRADING=false`, `SCALP_SESSIONS=...`, ultra_fast SL/TP/hold.
2. Run `.\start_mnq_forever.bat` (or `.\start_mnq_live.bat`).
3. Confirm banner: `PROFIT MODE: hybrid ultra_fast` and `Profit mode defaults applied`. Duration 0 = until stopped.
4. Trade **weekday RTH** in the scalp windows (≈09:30–12:00 and 13:30–16:00 ET).
5. Expect roughly **2–8 entries/hour** in liquid windows when trend+trigger align (not every minute).
6. After 20–40 live closes, review `data/trade_journal.jsonl` — if PF &lt; 1.0 on RTH only, pause and re-backtest.

## Backtest without Rithmic

You do **not** need a Rithmic plan to validate the recipe. Local CSVs in `data\` are enough:

```bat
.\start_mnq_backtest.bat
```

Or:

```bat
python scripts/backtest_scalp_hybrid.py --ultra-fast --rth-windows
```

`--rth-windows` matches live (09:30–12:00 and 13:30–16:00 ET).  
Optional: `.\start_mnq_backtest.bat rithmic-csv` uses `data/MNQ_1m_rithmic.csv` (already downloaded history, still offline).

Target: **PF &gt; 1.2** with commissions (all-hours winner historically ~2.2+; RTH-only will print fewer trades).
