# Live chart (READ-ONLY) — EVAL PASSER / FUNDED 12K
Local web page that shows the bot's trades on a moving chart. It never imports the bot and never talks to Rithmic: it only READS files.

Data sources (all on his PC, written by the bot):
1. 1m bars: the NEAT bar export CSV the bot already reads (`MNQ_1m_live.csv`, columns datetime,open,high,low,close,volume).
2. Trade events: NEW opt-in JSONL `logs\flex3_events.jsonl` (eval) / `logs\funded12k_events.jsonl` (funded), written only when
   env `FLEX3_EVENT_LOG` is set. Events: `entry` (side, setup, entry, SL, TP), `sl_move` (every stop change, incl. the $ lock), `exit` (price, reason, $ P&L).
   Hook = `src/strategy/flex3_events.py` + 4 one-line `F3EV.emit(...)` calls in the patch. Errors are swallowed; trading logic unchanged.
3. State: `logs\flex3_state.json` / `logs\funded12k_state.json` (day P&L, eval P&L / equity) the bot already writes.

Run: `START_LIVE_CHART.bat` (eval, :8765) · `START_LIVE_CHART.bat funded` (:8766) · `START_LIVE_CHART.bat replay` (demo 2026-09-15).
Replay (box): `python3 server.py --replay 2026-09-15 --speed 60` -> http://127.0.0.1:8765 . Needs Python 3 + pandas only; chart JS is vendored in static/.
