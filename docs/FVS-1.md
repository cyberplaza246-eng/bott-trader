# FVS-1 — Fabio Valentini Scalper (Rithmic / MNQ)

Rithmic-adapted implementation of Fabio Valentini's auction-market scalper. Uses existing 30s/1M bars and optional tick flow from `TickFlowTracker`. **Not** ccxt/crypto — built for CME micro futures via Rithmic.

## Concept

Triple-A framework: **A**uction state → **A**bsorption → **A**ggression.

| Layer | Name | Requirement |
|-------|------|-------------|
| 0 | Market state | Out of balance, impulse leg, ADX ≥ 25 |
| 1 | Volume profile | VP on impulse leg: POC, 70% value area, VAL, VAH, LVN |
| 2 | Absorption | 2+ bars at VAL (long) or VAH (short) — high vol, narrow range |
| 3 | Aggression | 30s break from absorption zone; structural SL/TP; 90s max hold |

## Live usage

```bat
set SCALP_MODE=fvs1_log
python start_live_mtf_scalping.py --symbols MNQ --paper
```

- `SCALP_MODE=fvs1_log` — diagnostics only (default `FVS1_LOG_ONLY=true`)
- `SCALP_MODE=fvs1` + `FVS1_LOG_ONLY=false` — live entries when all gates pass
- Coexists with hybrid: `hybrid` \| `fvs1` \| `fvs1_log`

Gate output matches hybrid style:

```
[PASS] GLOBAL: FVS1 session active (ny_open)
[PASS] LONG: [L0] out of balance + impulse ...
[PASS] LONG: [L1] VP POC=... VAL=... VAH=...
...
[BLOCK] SHORT: [L2] no absorption at VAH ...
```

Outside session:

```
[BLOCK] GLOBAL: outside FVS1 session (outside FVS1 session)
```

## Structural exits

| Side | Stop | Target |
|------|------|--------|
| Long | Below absorption zone (VAL area) | VAH |
| Short | Above absorption zone (VAH area) | VAL |

`MAX_HOLD_SECONDS=90` (env `FVS1_MAX_HOLD_SECONDS`). Round-trip fees ~$1–2 MNQ (`FVS1_ROUND_TRIP_FEE`).

## Session filter

Default windows (ET, Mon–Fri):

- **NY open** 09:30–10:00
- **PM** 14:00–15:00

Configure via `config/fvs1_sessions.yaml` or:

```
FVS1_SESSIONS=ny_open=09:30-10:00;pm=14:00-15:00
```

## Risk / safety

| Guard | Env | Default |
|-------|-----|---------|
| Consecutive loss halt | `FVS1_MAX_CONSECUTIVE_LOSSES` | 3 |
| Expectancy monitor | `FVS1_EXPECTANCY_WINDOW` | 50 trades |
| Pause on negative E | `FVS1_PAUSE_ON_NEGATIVE_E` | false |

Plus existing broker position block and `MAX_POSITIONS=2`.

## Backtest

```bat
python scripts/backtest_fvs1.py
python scripts/backtest_fvs1.py --csv data/MNQ_1m.csv --fee 1.50
```

Results: `data/fvs1_backtest_results.json`

## Key env vars

See `.env.example` section `FVS-1`. Opt-in via `SCALP_MODE=fvs1_log` — does not change hybrid defaults.
