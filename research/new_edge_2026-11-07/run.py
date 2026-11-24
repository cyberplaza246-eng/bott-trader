r"""Reproduce the whole study:  venv\Scripts\python.exe research\new_edge_2026-11-07\run.py
Writes everything into research\new_edge_2026-11-07\results\."""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E  # noqa: E402

OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

MAIN = r"C:\Users\shawa\Projects\bott-trader\data\MNQ_1m.csv"
H5 = r"C:\Users\shawa\Projects\bott-trader\research\live_audit_2026-11-07\NQ_5m_yahoo.csv"
H1 = r"C:\Users\shawa\Projects\bott-trader\research\live_audit_2026-11-07\NQ_1m_yahoo.csv"

# ----------------------------------------------------------------------------- locked design
SPLITS = {
    "TRAIN": ("2025-01-02", "2025-12-31"),
    "VALIDATE": ("2026-01-02", "2026-06-30"),
    "TEST": ("2026-07-01", "2026-09-21"),
    "HOLDOUT": ("2026-09-22", "2026-11-06"),
}
INSAMPLE = ("2025-01-02", "2026-09-21")
HOLDOUT_1M_START = "2026-10-12"  # ORB needs 1m bars; Yahoo 1m starts with the 2026-10-12 session

# 6 rolling folds, 3-month test windows stepping 3 months. Only 5 folds fit a full 6-month train
# inside 2025-01..2026-09, so fold 1 uses the available 3-month train (Jan-Mar 2025).
FOLDS = [
    (("2025-01-02", "2025-03-31"), ("2025-04-01", "2025-06-30")),
    (("2025-01-02", "2025-06-30"), ("2025-07-01", "2025-09-30")),
    (("2025-04-01", "2025-09-30"), ("2025-10-01", "2025-12-31")),
    (("2025-07-01", "2025-12-31"), ("2026-01-01", "2026-03-31")),
    (("2025-10-01", "2026-03-31"), ("2026-04-01", "2026-06-30")),
    (("2026-01-01", "2026-06-30"), ("2026-07-01", "2026-09-21")),
]

# strategy -> list of (cell_id, params). Every grid <= 12 cells, declared before any run.
GRID = {
    "S1_ORB": [(f"S1_ORB_r{r}_t{t}", dict(range_min=r, target_r=t)) for r in (15, 30) for t in (1.5, 2.0)],
    "S2_VWAP_MR": [(f"S2_VWAP_k{k}", dict(k=k)) for k in (1.5, 2.0)],
    "S3_PULLBACK": [(f"S3_PB_t{t}", dict(target_r=t)) for t in (1.5, 2.0)],
    "S4_MOMENTUM": [(f"S4_MOM_m{m}", dict(m=m)) for m in (0.25, 0.35)],
    "S2_VWAP_MR+RANGE": [(f"S2R_VWAP_k{k}", dict(k=k, regime="range")) for k in (1.5, 2.0)],
    "S3_PULLBACK+TREND": [(f"S3T_PB_t{t}", dict(target_r=t, regime="trend")) for t in (1.5, 2.0)],
    "S4_MOMENTUM+TREND": [(f"S4T_MOM_m{m}", dict(m=m, regime="trend")) for m in (0.25, 0.35)],
}
SLIPS = (1, 3)


def build_signals(strategy: str, params: dict, x1, x5, f5):
    if strategy == "S1_ORB":
        return E.sig_orb(x1, **params)
    if strategy.startswith("S2"):
        return E.sig_vwap_mr(x5, f5, **params)
    if strategy.startswith("S3"):
        return E.sig_pullback(x5, f5, **params)
    return E.sig_momentum(x5, f5, **params)


def main() -> None:
    t0 = time.time()
    data = E.load_all(MAIN, H5, H1, SPLITS["HOLDOUT"][0])
    log = data["log"]
    x1 = E.rth(data["m1"])
    x1 = x1[x1["day"] <= pd.Timestamp(INSAMPLE[1])]
    xh1 = E.rth(data["h1"])
    xh1 = xh1[xh1["day"] >= pd.Timestamp(HOLDOUT_1M_START)]
    x5 = E.rth(data["m5_comb"])
    f5 = E.features_5m(x5)
    log.update(rth_1m_bars_insample=len(x1), rth_5m_bars_all=len(x5), rth_1m_bars_holdout=len(xh1),
               insample_days=int(x1["day"].nunique()),
               holdout_days_5m=int(x5[x5["day"] >= pd.Timestamp(SPLITS["HOLDOUT"][0])]["day"].nunique()),
               holdout_days_1m=int(xh1["day"].nunique()))
    print("data:", json.dumps(log, indent=1, default=str))

    runs: dict[tuple[str, int], E.RunResult] = {}
    hold_runs: dict[tuple[str, int], E.RunResult] = {}
    cell_strategy = {}
    all_trades = []
    for strat, cells in GRID.items():
        for cid, params in cells:
            cell_strategy[cid] = strat
            sig = build_signals(strat, params, x1, x5, f5)
            sig_h = E.sig_orb(xh1, **params) if strat == "S1_ORB" else None
            for slip in SLIPS:
                if strat == "S1_ORB":
                    rr = E.simulate(x1, sig, slip)
                    rh = E.simulate(xh1, sig_h, slip)
                else:
                    full = E.simulate(x5, sig, slip)
                    rr = E.slice_run(full, *INSAMPLE)
                    rh = E.slice_run(full, *SPLITS["HOLDOUT"])
                runs[(cid, slip)] = rr
                hold_runs[(cid, slip)] = rh
                for tag, r in (("insample", rr), ("holdout", rh)):
                    if len(r.trades):
                        t = r.trades.copy()
                        t.insert(0, "slip_ticks", slip)
                        t.insert(0, "cell", cid)
                        t.insert(0, "strategy", strat)
                        t["segment"] = tag
                        all_trades.append(t)
            print(f"  ran {cid}  ({time.time() - t0:.0f}s)")

    def split_run(cid, slip, name):
        if name == "HOLDOUT":
            return hold_runs[(cid, slip)]
        return E.slice_run(runs[(cid, slip)], *SPLITS[name])

    # ---------------------------------------------------------------- sensitivity: every cell
    sens_rows, cell_stats = [], {}
    for cid, strat in cell_strategy.items():
        for slip in SLIPS:
            for name in list(SPLITS) + ["INSAMPLE_ALL"]:
                rr = runs[(cid, slip)] if name == "INSAMPLE_ALL" else split_run(cid, slip, name)
                st = E.stats(rr)
                cell_stats[(cid, slip, name)] = st
                sens_rows.append(dict(strategy=strat, cell=cid, slip_ticks=slip, split=name,
                                      **{k: v for k, v in st.items() if k not in ("breaches", "pnl_by_entry_hour")}))
    sens = pd.DataFrame(sens_rows)
    sens.to_csv(os.path.join(OUT, "sensitivity_all_cells.csv"), index=False)

    # ---------------------------------------------------------------- per strategy
    summary, details, wf_rows, breach_rows = [], {}, [], []
    for strat, cells in GRID.items():
        ids = [c for c, _ in cells]
        # selection on TRAIN (1 tick), ties -> first declared cell
        sel = max(ids, key=lambda c: (cell_stats[(c, 1, "TRAIN")]["net"], -ids.index(c)))
        # walk-forward
        wf_total, wf_pos, fold_info = 0.0, 0, []
        for fi, (tr_w, te_w) in enumerate(FOLDS, 1):
            nets = {c: E.stats(E.slice_run(runs[(c, 1)], *tr_w))["net"] for c in ids}
            pick = max(ids, key=lambda c: (nets[c], -ids.index(c)))
            te = E.stats(E.slice_run(runs[(pick, 1)], *te_w))
            te3 = E.stats(E.slice_run(runs[(pick, 3)], *te_w))
            wf_total += te["net"]
            wf_pos += te["net"] > 0
            row = dict(strategy=strat, fold=fi, train=f"{tr_w[0]}..{tr_w[1]}", test=f"{te_w[0]}..{te_w[1]}",
                       picked=pick, train_net=nets[pick], test_net=te["net"], test_net_3tick=te3["net"],
                       test_trades=te["trades"], test_mdd_mtm=te["mdd_mtm"])
            wf_rows.append(row)
            fold_info.append(row)
        # continuous in-sample drawdown for the selected cell (train+validate+test)
        ins = cell_stats[(sel, 1, "INSAMPLE_ALL")]
        for b in ins["breaches"]:
            breach_rows.append(dict(strategy=strat, cell=sel, scope="INSAMPLE_ALL(selected)", **b))
        for c in ids:  # every cell, every split, so failing configs are visible
            for name in list(SPLITS) + ["INSAMPLE_ALL"]:
                for b in cell_stats[(c, 1, name)]["breaches"]:
                    breach_rows.append(dict(strategy=strat, cell=c, scope=name, **b))
        s = {n: cell_stats[(sel, 1, n)] for n in SPLITS}
        s3 = {n: cell_stats[(sel, 3, n)] for n in SPLITS}
        crit = {
            "validate_net>0": s["VALIDATE"]["net"] > 0,
            "test_net>0": s["TEST"]["net"] > 0,
            "wf_total>0": wf_total > 0,
            "wf_folds_pos>=4": wf_pos >= 4,
            "3tick_validate&test>0": s3["VALIDATE"]["net"] > 0 and s3["TEST"]["net"] > 0,
            "no_2k_breach_insample": ins["breach_count"] == 0,
        }
        candidate = all(crit.values())
        summary.append(dict(
            strategy=strat, selected_cell=sel, cells=len(ids),
            train_net=s["TRAIN"]["net"], validate_net=s["VALIDATE"]["net"], test_net=s["TEST"]["net"],
            wf_oos_total=wf_total, wf_folds_pos=f"{wf_pos}/6", holdout_net=s["HOLDOUT"]["net"],
            holdout_trades=s["HOLDOUT"]["trades"],
            validate_net_3t=s3["VALIDATE"]["net"], test_net_3t=s3["TEST"]["net"], holdout_net_3t=s3["HOLDOUT"]["net"],
            insample_net=ins["net"], insample_mdd_mtm=ins["mdd_mtm"], insample_mdd_eod=ins["mdd_eod"],
            insample_breaches=ins["breach_count"], insample_trades=ins["trades"], insample_t=ins["t_stat_daily"],
            candidate=candidate, failed=";".join(k for k, v in crit.items() if not v)))
        details[strat] = dict(selected_cell=sel, criteria=crit, candidate=candidate, splits=s,
                              splits_3tick=s3, insample_all=ins, walk_forward=fold_info)

    summ = pd.DataFrame(summary)
    summ.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    pd.DataFrame(wf_rows).to_csv(os.path.join(OUT, "walkforward.csv"), index=False)
    pd.DataFrame(breach_rows, columns=["strategy", "cell", "scope", "date", "dd"]).to_csv(
        os.path.join(OUT, "dd_breaches.csv"), index=False)
    pd.concat(all_trades, ignore_index=True).to_csv(os.path.join(OUT, "trades_all.csv"), index=False)

    n_cells = sum(len(v) for v in GRID.values())
    # luck model: zero-edge strategy, VAL>0 and TEST>0 ~ coin flips, >=4/6 folds ~ Binomial(6,.5)
    p_luck = 0.5 * 0.5 * (22 / 64)
    meta = dict(data=log, n_strategies=len(GRID), n_cells=n_cells, n_simulations=n_cells * len(SLIPS),
                p_pass_by_luck_per_strategy=p_luck, expected_lucky_candidates=len(GRID) * p_luck,
                bonferroni_t_5pct_two_sided_over_cells=float(abs(np.round(
                    __import__("statistics").NormalDist().inv_cdf(0.025 / n_cells), 2))),
                holdout_note=f"S1_ORB holdout uses 1m bars from {HOLDOUT_1M_START}; S2-S4 use 5m bars "
                             f"for all splits, holdout {SPLITS['HOLDOUT'][0]}..{SPLITS['HOLDOUT'][1]}",
                folds_note="fold 1 has only a 3-month train window (no data before 2025)",
                runtime_s=round(time.time() - t0, 1))
    with open(os.path.join(OUT, "details.json"), "w") as fh:
        json.dump(dict(meta=meta, strategies=details), fh, indent=1, default=str)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    cols = ["strategy", "selected_cell", "train_net", "validate_net", "test_net", "wf_oos_total", "wf_folds_pos",
            "holdout_net", "holdout_trades", "validate_net_3t", "test_net_3t", "insample_mdd_mtm",
            "insample_breaches", "candidate", "failed"]
    print("\nSUMMARY (selected-on-TRAIN cell, 1 tick)\n", summ[cols].round(0).to_string(index=False))
    sv = sens[(sens.slip_ticks == 1)].pivot_table(index="cell", columns="split", values="net", sort=False)
    print("\nSENSITIVITY net $ (1 tick, every cell)\n", sv.round(0).to_string())
    print("\nmeta:", json.dumps({k: v for k, v in meta.items() if k != "data"}, indent=1))


if __name__ == "__main__":
    main()
