# Backtests (2026-10-08 snapshot)

Stopped mid-run at the user's request. This folder is the work finished so far. No live bot code was changed.

## Folders

### `nq_rsi/`
Revamp strategy backtested on **MNQ vs NQ**, plus an RSI-50 variant experiment.

**NQ vs MNQ revamp** (1 contract, Nov 2024–Sep 2026, `$500` daily cap / dollar lock settings as labeled in `out/summary.csv`):

| | MNQ | NQ |
|---|---|---|
| Net profit | +$15,835 | +$166,726 |
| Profit factor | 1.36 | 1.41 |
| Max drawdown | −$3,921 | −$29,300 |

NQ is unsuitable for a `$500` daily loss cap: typical stop width ~124 pts ≈ `$2,500` per NQ contract. See also `out/stops.json` and the full grid in `out/summary.csv`.

**RSI-50 variant** (sell below 50 / buy above 50 style): lost money. Pure RSI-50 rule ≈ **−$10,533** vs baseline **+$8,194** over the same MNQ window (`out/rsi_variants.csv`: `B_none_atr1.5_x70` vs `A_baseline`). Trend-filtered RSI was about flat / still weak on holdout. Leave the live strategy as-is.

Contents: `code/*.py`, small `out/` csv/json/log (skipped `rsi_trades.pkl` >5MB).

### `revamp_vs_visible/` — PARTIAL
Attempt to rebuild the OG Visible bot for a fair head-to-head vs the revamp on the same MNQ data. **Stopped before finishing**; only signal/debug scripts and a tiny test pickle are here. No completed comparison results.

### `neat_v2/` — PARTIAL
NEAT “kid” training v2 aimed at more terrains (trending / chop / volatile MNQ). **Stopped before finishing** a fair holdout bake-off.

Included: training/eval scripts, `neat.cfg`, `norm.json`, `holdout_old.json` (old champions cycle 78 / 130 on Apr–Sep style holdout — both still losing on that window), per-island `runs/*/params.json` + `train.log` + small `hof.pkl`. Skipped `feat5m.parquet` (~40MB), `pylib/`, `__pycache__/`.

## Not included
- Secrets, `.env`, credentials
- Large data / pickle dumps (>5MB)
- Live launcher or bot source changes
