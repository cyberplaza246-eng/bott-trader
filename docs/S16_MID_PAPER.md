# MNQ S16 mid paper (Lucid TEST)

Runs the locked **HybridS15_IB** recipe (ORB 20 + IB 45, long-only, target_r=0.75) as a continuous bot on Lucid **TEST** paper. It replaces day hybrid for paper. It does **not** trade funded live money.

Lock fingerprint: `e8ad933573b8bbf3` (do not edit `data\MNQ_PASS_s16_mid.json`).

## Before you start

1. Close Lucid desktop / R|Trader if it is open (one Lucid login only).
2. Stop any other bot that uses Lucid:
   - day hybrid (`start_mnq_day_hybrid.bat`)
   - overnight paper
   - micro-scalp paper
3. Confirm `.env` has `TRADING_MODE=paper` and a TEST account id (e.g. `...TEST012`).

## Start

```bat
start_mnq_s16_mid_paper.bat
```

Optional size (default 1):

```bat
set S16_MID_SIZE=1
start_mnq_s16_mid_paper.bat
```

Smoke (connect then exit after 1 minute):

```bat
venv\Scripts\python.exe -u scripts\run_s16_mid_paper.py --duration 1
```

## Stop

Focus the bot window and press **Ctrl+C**. The bot flattens MNQ on the sim and shuts down the Lucid session.

## What you should see

- Banner says **S16 MID PAPER** / session `s16_mid` (not `day_hybrid`)
- Lock fingerprint `e8ad933573b8bbf3`
- Lucid TEST connect + MNQ quote / heartbeats like `flat | RTH | scanning ORB/IB`
- `live_mode=False` and trade route `simulator`

## Safety

- `ALLOW_LIVE=false` — funded live stays off
- Paper brackets only
- Flatten near RTH close (~15:55 ET)
- Daily loss halt 2% (from lock)
- Max 10 contracts hard cap; start size 1
- No scale-in, no trail

## Files

- `src\strategy\s16_mid.py` — locked signal logic
- `src\strategy\s16_mid_paper_runner.py` — Lucid paper loop
- `scripts\run_s16_mid_paper.py` — CLI
- `data\MNQ_PASS_s16_mid.json` — lock (read-only; do not edit)
- `start_mnq_s16_mid_paper.bat` — start script
