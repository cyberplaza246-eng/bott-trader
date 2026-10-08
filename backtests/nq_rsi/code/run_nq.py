"""NQ vs MNQ backtest of the REVAMP 5m ensemble (SIMULATION ONLY, no broker/network)."""
import json, sys
import numpy as np, pandas as pd
from data import load_1m, resample_5m
from signals import build_signals, live_atr_sma14
import engine as E

OUT = "/workspace/nq_bt/out"
START = pd.Timestamp("2024-11-01", tz="America/New_York")
HOLD = pd.Timestamp("2026-04-01", tz="America/New_York")
END = pd.Timestamp("2026-10-01", tz="America/New_York")
ONE_MIN = 60_000_000_000
INSTR = {  # pv $/pt, round-trip cost per contract = commission + 1 tick slippage per side
    "MNQ": dict(pv=2.0, comm=2.00, tick_usd=0.50),
    "NQ": dict(pv=20.0, comm=4.50, tick_usd=5.00),
}
for v in INSTR.values():
    v["cost"] = v["comm"] + 2 * v["tick_usd"]

def conf(r):
    if r["sleeve"] == "mean_reversion" or not np.isfinite(r["adx"]):
        return 0.66
    return max(0.5, min(0.95, 0.62 + (r["adx"] - 20.0) * 0.01))

def size_for(c, sizing):
    if not sizing: return 1
    return 4 if c >= 0.85 else 3 if c >= 0.78 else 2 if c >= 0.72 else 1

def gate(att, pv, cost, cfg):
    att = att.sort_values("T").reset_index(drop=True).copy()
    att["day"] = pd.to_datetime(att["T"], utc=True).dt.tz_convert("America/New_York").dt.date
    att["conf"] = att.apply(conf, axis=1)
    att["size"] = att["conf"].apply(lambda c: size_for(c, cfg["sizing"]))
    att["pnl"] = (att["pts"] * pv - cost) * att["size"]
    att["risk_usd"] = (att["fill"] - att["sl"]).abs() * pv * att["size"]
    taken = np.zeros(len(att), bool)
    acc = []  # (start, end, exit_time, pnl, day, filled)
    s_ns, e_ns = START.value, END.value
    for i, r in att.iterrows():
        T = r["T"]
        if T < s_ns or T >= e_ns: continue
        day = r["day"]
        busy = sum(1 for a in acc if a[0] <= T < a[1])
        if busy >= cfg["max_conc"]: continue
        if busy >= 1 and r["conf"] < cfg["extra_conf"]: continue
        done = sorted([a for a in acc if a[5] and a[2] <= T], key=lambda a: a[2])
        if done:
            last = done[-1]
            cd = cfg["cd_win"] if last[3] > 0 else cfg["cd_loss"]
            if T - last[2] < cd * ONE_MIN: continue
        today = [a for a in done if a[4] == day]
        if sum(a[3] for a in today) <= -cfg["daily_loss"]: continue
        cnt, pu = 0, -1
        for a in today:
            if pu >= 0 and a[2] >= pu: cnt, pu = 0, -1
            if a[3] < 0:
                cnt += 1
                if cnt >= 3: pu = a[2] + 30 * ONE_MIN
            elif a[3] > 0: cnt = 0
        if pu >= 0 and T < pu: continue
        if r["status"] == "filled":
            acc.append((T, r["slot_end"], r["exit_time"], r["pnl"], day, True))
        else:
            acc.append((T, r["slot_end"], None, 0.0, day, False))
        taken[i] = True
    att["taken"] = taken
    return att

def metrics(g, lo, hi):
    t = g[g["taken"] & (g["status"] == "filled")]
    t = t[(t["T"] >= lo.value) & (t["T"] < hi.value)].sort_values("exit_time")
    p = t["pnl"].to_numpy(); eq = np.cumsum(p)
    dd = float((eq - np.maximum.accumulate(np.r_[0.0, eq])[1:]).min()) if len(p) else 0.0
    gw, gl = p[p > 0].sum(), -p[p < 0].sum()
    mon = pd.to_datetime(t["exit_time"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.to_period("M")
    months = pd.period_range(lo.tz_localize(None), (hi - pd.Timedelta(days=1)).tz_localize(None), freq="M")
    m = t.groupby(mon.values)["pnl"].sum().reindex(months, fill_value=0.0)
    dayp = t.groupby("day")["pnl"].sum()
    return dict(net=round(p.sum(), 0), trades=len(p), win_rate=round(100 * (p > 0).mean(), 1) if len(p) else np.nan,
                pf=round(gw / gl, 2) if gl else np.inf, max_dd=round(dd, 0), avg_month=round(m.mean(), 0),
                worst_month=f"{m.idxmin()} {m.min():.0f}", green_months=f"{(m > 0).sum()}/{len(m)}",
                worst_day=round(dayp.min(), 0) if len(dayp) else 0, days_below_500=int((dayp <= -500).sum()),
                days_below_400=int((dayp <= -400).sum()))

def main():
    d1 = load_1m(); df5 = resample_5m(d1)
    sigs = build_signals(df5)
    sigs = sigs[(sigs["signal_time"] >= START) & (sigs["signal_time"] < END)]
    B = E.Bars(d1, df5, live_atr_sma14(df5))
    CFGS = {
        # identical risk rules to the last MNQ revamp study (except instrument)
        "last_study": dict(lock="usd_old", max_conc=2, extra_conf=0.0, cd_win=25, cd_loss=25, daily_loss=200, sizing=False),
        # last study rules but Lucid $500 daily cap
        "last_study_dl500": dict(lock="usd_old", max_conc=2, extra_conf=0.0, cd_win=25, cd_loss=25, daily_loss=500, sizing=False),
        # current start_mnq_live_revamp.bat: R-lock, 3 concurrent (extras conf>=0.72), cooldown 1 bar win / 15 min loss, $500 daily
        "live_now_1lot": dict(lock="r", max_conc=3, extra_conf=0.72, cd_win=5, cd_loss=15, daily_loss=500, sizing=False),
        "live_now_1to4": dict(lock="r", max_conc=3, extra_conf=0.72, cd_win=5, cd_loss=15, daily_loss=500, sizing=True),
    }
    rows, trades = [], []
    cache = {}
    for cname, cfg in CFGS.items():
        for inst, sp in INSTR.items():
            E.LOCK_MODE = cfg["lock"]; E.PV = sp["pv"]
            key = (cfg["lock"], sp["pv"] if cfg["lock"] == "usd_old" else 0)
            if key not in cache:
                cache[key] = E.attempts(B, sigs, "base", False)
            g = gate(cache[key], sp["pv"], sp["cost"], cfg)
            for per, lo, hi in (("full_2024-11_2026-09", START, END), ("holdout_2026-04_2026-09", HOLD, END)):
                m = metrics(g, lo, hi); m.update(config=cname, instr=inst, period=per); rows.append(m)
            t = g[g["taken"] & (g["status"] == "filled")].copy(); t["config"], t["instr"] = cname, inst
            trades.append(t)
    S = pd.DataFrame(rows); T = pd.concat(trades)
    import os; os.makedirs(OUT, exist_ok=True)
    S.to_csv(f"{OUT}/summary.csv", index=False); T.to_csv(f"{OUT}/trades.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(S[["config", "instr", "period", "net", "trades", "win_rate", "pf", "max_dd", "avg_month", "worst_month", "green_months", "worst_day", "days_below_500", "days_below_400"]].to_string(index=False))
    # stop size distribution (initial stop, 1 contract) over all filled signals of live_now_1lot
    t = T[(T.config == "live_now_1lot") & (T.instr == "MNQ")]
    pts = (t["fill"] - t["sl"]).abs()
    q = pts.quantile([.1, .25, .5, .75, .9, .95]).round(2)
    out = dict(stop_pts_quantiles=q.to_dict(), n=len(pts), mean_pts=round(pts.mean(), 2),
               nq_stop_usd_median=round(pts.median() * 20 + 14.5, 0),
               share_nq_stop_ge_500=round(float(((pts * 20 + 14.5) >= 500).mean()), 3),
               share_nq_stop_ge_400=round(float(((pts * 20 + 14.5) >= 400).mean()), 3),
               share_nq_stop_ge_250=round(float(((pts * 20 + 14.5) >= 250).mean()), 3),
               by_sleeve_median_pts=t.assign(p=pts).groupby("sleeve")["p"].median().round(2).to_dict(),
               by_sleeve_n=t["sleeve"].value_counts().to_dict(),
               sizing_dist=T[(T.config == "live_now_1to4") & (T.instr == "MNQ")]["size"].value_counts().sort_index().to_dict())
    hold = t[t["T"] >= HOLD.value]; hp = (hold["fill"] - hold["sl"]).abs()
    out.update(holdout_median_pts=round(hp.median(), 2), holdout_share_nq_ge_500=round(float(((hp * 20 + 14.5) >= 500).mean()), 3),
               holdout_share_nq_ge_400=round(float(((hp * 20 + 14.5) >= 400).mean()), 3))
    lo = T[(T.config == "live_now_1lot") & (T.instr == "NQ") & (T.pnl < 0)]
    out.update(nq_losers=len(lo), nq_loser_median=round(lo.pnl.median(), 0), nq_losers_le_m500=int((lo.pnl <= -500).sum()),
               nq_losers_le_m400=int((lo.pnl <= -400).sum()))
    print(json.dumps(out, indent=1, default=str))
    json.dump(out, open(f"{OUT}/stops.json", "w"), indent=1, default=str)

if __name__ == "__main__":
    main()
