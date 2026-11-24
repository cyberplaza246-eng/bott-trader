"""Standalone event/time add-on tests on MNQ 1m (BACKTEST ONLY). Costs: $4 RT + 1 tick slip each side. No lookahead: signals use bars closed before entry."""
import os,sys,json
os.environ['PYTHONDONTWRITEBYTECODE']='1'; sys.dont_write_bytecode=True
R=os.path.dirname(os.path.abspath(__file__))
import sr_engine as S
E=S.E; import numpy as np, pandas as pd
ET=S.ET; PV=2.0; COMM=4.0; SLIP=0.25
d=E.load(); m1=d['m1']
o=m1['open']; c=m1['close']; h=m1['high']; l=m1['low']
def px_open_at(ts):
    i=m1.index.searchsorted(ts)           # first bar at/after ts
    if i>=len(m1): return None,None
    return i,m1.index[i]
def trade(ts_in,ts_out,direction,maxgap_min=3):
    i,t=px_open_at(ts_in); j,u=px_open_at(ts_out)
    if i is None or j is None: return None
    if (t-ts_in)>pd.Timedelta(minutes=maxgap_min) or (u-ts_out)>pd.Timedelta(minutes=maxgap_min): return None
    ent=o.iloc[i]+direction*SLIP; ex=o.iloc[j]-direction*SLIP
    return dict(entry_time=t,exit_time=u,dir=direction,pnl=direction*(ex-ent)*PV-COMM)
def at(ts):    # open price of first bar at/after ts
    i,t=px_open_at(ts); 
    if i is None or (t-ts)>pd.Timedelta(minutes=3): return np.nan
    return o.iloc[i]
def ETts(day,hm): return pd.Timestamp(f'{day} {hm}',tz=ET).tz_convert('UTC')
out={}
def rec(name,rows):
    df=pd.DataFrame([r for r in rows if r]); 
    if len(df): df.to_csv(os.path.join(R,'addon_'+name+'.csv'),index=False)
    out[name]=df; return df
# ---- A1 last-30-min momentum (Gao 2018 / Baltussen 2021): sign(price@15:30 - prior RTH close@16:00) -> hold 15:30..15:59
days=pd.bdate_range('2024-10-01','2026-09-18')
rows=[];prev=None
for dd in days:
    ds=dd.strftime('%Y-%m-%d'); pc=at(ETts(ds,'16:00')) if False else None
    p_close=m1['close'].asof(ETts(ds,'15:59'))  # close of 15:59 bar ~ 16:00 price (bar stamped <=15:59)
    if prev is not None and np.isfinite(prev):
        p1530=at(ETts(ds,'15:30'))
        if np.isfinite(p1530):
            sg=np.sign(p1530-prev)
            if sg!=0:
                rows.append(trade(ETts(ds,'15:30'),ETts(ds,'15:58'),int(sg)))
    prev=p_close if abs(m1.index.asof(ETts(ds,'15:59'))-ETts(ds,'15:59'))<pd.Timedelta(minutes=5) else np.nan
rec('A1_last30_mom',rows)
rows=[r.copy() if r else None for r in rows]
for r in rows:
    if r: r['dir']*=-1; 
# fade version: recompute pnl from scratch
rows2=[]
prev=None
for dd in days:
    ds=dd.strftime('%Y-%m-%d'); p_close=m1['close'].asof(ETts(ds,'15:59'))
    if prev is not None and np.isfinite(prev):
        p1530=at(ETts(ds,'15:30'))
        if np.isfinite(p1530) and p1530!=prev: rows2.append(trade(ETts(ds,'15:30'),ETts(ds,'15:58'),-int(np.sign(p1530-prev))))
    prev=p_close if abs(m1.index.asof(ETts(ds,'15:59'))-ETts(ds,'15:59'))<pd.Timedelta(minutes=5) else np.nan
rec('A1b_last30_fade',rows2)
# ---- A2 pre-FOMC drift long: 14:00 day before announcement -> 13:55 announcement day
rows=[]
for f in S.FOMC:
    day=pd.Timestamp(f); prevd=(day-pd.offsets.BDay(1)).strftime('%Y-%m-%d')
    rows.append(trade(ETts(prevd,'14:00'),ETts(f,'13:55'),1,maxgap_min=10))
rec('A2_preFOMC_long',rows)
# ---- A3 event spike: first 5 min move after release; enter at +5, hold 30 min; continuation vs fade
def spike(events_dates,hm,tag,hold=30):
    rc=[];rf=[];big_c=[];big_f=[];mv=[]
    recs=[]
    for dstr in events_dates:
        t0=ETts(dstr,hm); a=at(t0); b=at(t0+pd.Timedelta(minutes=5))
        if not(np.isfinite(a) and np.isfinite(b)): continue
        recs.append((dstr,t0,b-a))
    if not recs: return
    mags=np.array([abs(r[2]) for r in recs]); med=np.median(mags)   # NB median uses all events (descriptive threshold only for 'big' split)
    for dstr,t0,m in recs:
        if m==0: continue
        sg=int(np.sign(m)); tin=t0+pd.Timedelta(minutes=5); tout=tin+pd.Timedelta(minutes=hold)
        rc.append(trade(tin,tout,sg)); rf.append(trade(tin,tout,-sg))
        if abs(m)>=med: big_c.append(trade(tin,tout,sg)); big_f.append(trade(tin,tout,-sg))
    rec(f'A3_{tag}_cont',rc); rec(f'A3_{tag}_fade',rf); rec(f'A3_{tag}_bigcont',big_c); rec(f'A3_{tag}_bigfade',big_f)
spike(S.NFP,'08:30','NFP'); spike(S.CPI,'08:30','CPI'); spike(S.FOMC,'14:00','FOMC')
spike(S.NFP+S.CPI,'08:30','NFPCPI'); 
# ---- A4 overnight gap at 9:30 (RTH open vs prior 15:59 close): continuation vs fade, hold to 10:30; trade only if |gap| >= expanding median of past |gaps| (no lookahead)
rows_c=[];rows_f=[];past=[];prev=None
for dd in days:
    ds=dd.strftime('%Y-%m-%d'); o930=at(ETts(ds,'09:30'))
    if prev is not None and np.isfinite(prev) and np.isfinite(o930):
        gap=o930-prev
        if len(past)>=20 and abs(gap)>=np.median(past) and gap!=0:
            sg=int(np.sign(gap)); rows_c.append(trade(ETts(ds,'09:30'),ETts(ds,'10:30'),sg)); rows_f.append(trade(ETts(ds,'09:30'),ETts(ds,'10:30'),-sg))
        past.append(abs(gap))
    pcl=m1['close'].asof(ETts(ds,'15:59')); prev=pcl if abs(m1.index.asof(ETts(ds,'15:59'))-ETts(ds,'15:59'))<pd.Timedelta(minutes=5) else np.nan
rec('A4_gap_cont',rows_c); rec('A4_gap_fade',rows_f)
# ---- summary
cut=pd.Timestamp('2026-01-01',tz='UTC'); summ=[]
for k,df in out.items():
    if len(df)==0: continue
    p=df.pnl.values; g=p[p>0].sum(); ls=-p[p<0].sum()
    tr=df[df.entry_time<cut].pnl; te=df[df.entry_time>=cut].pnl
    t=p.mean()/(p.std(ddof=1)/np.sqrt(len(p))) if len(p)>2 and p.std(ddof=1)>0 else 0
    summ.append(dict(name=k,n=len(p),net=round(p.sum(),0),avg=round(p.mean(),1),win=round(100*(p>0).mean(),1),pf=round(g/ls,2) if ls>0 else 99,t=round(t,2),train=round(tr.sum(),0),n_tr=len(tr),y2026=round(te.sum(),0),n_26=len(te)))
json.dump(summ,open(os.path.join(R,'addons_summary.json'),'w'),indent=1)
for r in summ: print(r)
