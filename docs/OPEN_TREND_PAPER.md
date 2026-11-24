# OpenTrend MNQ paper bot

OpenTrend is a small, open-rules **paper** bot for MNQ. It is an original implementation and is **not Trend Catcher**, does not clone Trend Catcher, and makes no claim to reproduce that product.

## Rules (v1 locked)

- Symbol: MNQ, one contract, one position at a time, no pyramiding.
- Timeframe: completed 5-minute bars from the same CSV/Databento and Lucid/Rithmic history path used by the S16 mid paper runner.
- Indicators: EMA(9), EMA(21), and ATR(14).
- Long: on a completed bar, EMA(9) crosses above EMA(21) and the close is above EMA(21).
- Short: on a completed bar, EMA(9) crosses below EMA(21) and the close is below EMA(21).
- Stop loss: 1.5 x ATR(14) from the entry fill.
- Take profit: 2.0 x ATR(14) from the entry fill.
- Entries are allowed during approximately 09:30–15:55 America/New_York. The bot flattens its position at 15:55 ET.
- Orders are sent only to the configured Lucid TEST/simulator path. `ALLOW_LIVE=false` is forced by the CLI and launcher.

This is software for testing. It is not investment advice. Paper fills, history, brackets, and data freshness may differ from live trading. Do not enable funded live trading.

## Start

From the repository root, double-click:

```text
start_mnq_open_trend_paper.bat
```

Or from an activated repository virtual environment:

```bat
set TRADING_MODE=paper
set ALLOW_LIVE=false
set PAPER_USE_RITHMIC=true
set PAPER_RITHMIC_BRACKETS=true
python -u scripts\run_open_trend_paper.py
```

For a bounded smoke session (not a live session):

```bat
python -u scripts\run_open_trend_paper.py --duration 1
```

Close any other bot using the Lucid session first; the existing runners use one Lucid login at a time.

## Stop

Press `Ctrl+C` in the OpenTrend window. The runner attempts to close its own position and cancel its own protective orders before shutting down. The launcher also pauses after exit so the result can be read.

No funded-live mode is supported by this launcher. Keep the paper/test settings enabled.
