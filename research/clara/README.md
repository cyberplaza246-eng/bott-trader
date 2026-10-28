# Clara Core behavioral replica (public data only)
Data: nt300.csv/tv300.csv = Clara public trade logs (clarasystems.io/downloads.html); MNQ_1m.csv copied read-only from Paragan.
Reproduce:  python3 an.py && python3 feat.py && python3 h1.py && python3 fit.py && python3 report.py
Replica logic: sim.py run(P); chosen P in report.py (picked by fidelity to Clara's trades on 2024-09..2025-12, not by profit).
Outputs: report.out, replica_trades.csv, fit.csv, rea_perf.json (REA web-bundle evidence)
## Round 2 (filters, other bots, fusion)
python3 feat_day.py   # day features (pre-entry only), then the ybody/open_in_y/noise add-on is in final notes
python3 filt.py       # 120 train variants -> filt_train.csv
python3 wf.py && python3 wf2.py   # 168 morning variants over full period -> wf_all*.pkl
python3 sixpm.py      # 6 PM long w/ Monday skip + trail (+100pt -> high-3xATR15) -> six.pkl
python3 final.py      # train pick, walk-forward, 2026 test, 15 fusion combos -> final.out, combos.csv
others/pq.py, others/pq2.py  # Pinnacle Quant NQ log audit
