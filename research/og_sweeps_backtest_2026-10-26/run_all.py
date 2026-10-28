import sim, json, os, itertools, sys
import pandas as pd, numpy as np
H=os.path.dirname(os.path.abspath(__file__)); OUT=H
C=sim.load_signals(); F=sim.load_fine()
print('signals',len(C),'fine bars',len(F),flush=True)
sd=pd.DataFrame(C); sd['t']=pd.to_datetime(sd['t'],utc=True)

def tag(df):
    df=df.copy()
    et=df.entry_t.dt.tz_convert('America/New_York'); df['et']=et
    df['year']=et.dt.year; df['month']=et.dt.strftime('%Y-%m'); df['dow']=et.dt.dayofweek
    df['sunday']=np.where(df.dow==6,'Sunday','Mon-Sat'); df['hour_et']=et.dt.hour
    return df
def periods(df):
    d=df
    P={'ALL':d,'2024 (Sep22-Dec)':d[d.year==2024],'2025':d[d.year==2025],'2026 YTD':d[d.year==2026],
       '2026 Jan-Sep':d[(d.year==2026)&(d.month<='2026-09')],'2026 Oct (live bars)':d[d.month=='2026-10']}
    return P
def summ(df):
    out={}
    for k,v in periods(df).items():
        row={}
        for col,nm in (('pnl_gross','gross'),('pnl_bot','bot'),('pnl_real','real')):
            m=sim.metrics(v,col); row[nm]=m
        out[k]=row
    return out

base,openend=sim.simulate(C,F); base=tag(base)
base.to_csv(os.path.join(OUT,'base_trades.csv'),index=False)
S=summ(base); json.dump(S,open(os.path.join(OUT,'base_summary.json'),'w'),indent=1,default=str)
print('BASE',json.dumps({k:{c:{x:round(y,2) for x,y in m.items()} for c,m in v.items()} for k,v in S.items()},indent=0)[:3000],flush=True)
# monthly
mon=base.groupby('month').agg(n=('pnl_bot','size'),gross=('pnl_gross','sum'),bot=('pnl_bot','sum'),real=('pnl_real','sum')).round(0)
mon.to_csv(os.path.join(OUT,'base_monthly.csv'))
# long/short, sunday, reason, conf buckets, regime, hour
def grp(col):
    g=base.groupby(col).apply(lambda x: pd.Series(dict(n=len(x),wr=(x.pnl_bot>0).mean()*100,net_bot=x.pnl_bot.sum(),net_real=x.pnl_real.sum(),avg=x.pnl_bot.mean())),include_groups=False).round(1)
    return g
base['conf_b']=pd.cut(base.conf,[0,.64,.68,.72,.78,.85,1.01]).astype(str)
tabs={k:grp(k) for k in ['dir','sunday','reason','conf_b','regime','hour_et','size','year']}
with open(os.path.join(OUT,'base_breakdowns.txt'),'w') as f:
    for k,v in tabs.items(): f.write(f'--- {k}\n{v.to_string()}\n\n')
mon_str=mon.to_string()
# stats: SL/TP distances
sl_pts=(base.entry-base.sl0).abs(); tp_pts=(base.tp0-base.entry).abs()
rr=dict(avg_sl_pts=float(sl_pts.mean()),avg_tp_pts=float(tp_pts.mean()),median_sl=float(sl_pts.median()),median_tp=float(tp_pts.median()))
rr['breakeven_wr_pct_no_cost']=float(100*rr['avg_sl_pts']/(rr['avg_sl_pts']+rr['avg_tp_pts']))
rr['tp_hit_pct']=float((base.reason=='TAKE_PROFIT').mean()*100); rr['sl_hit_pct']=float(base.reason.str.startswith('STOP').mean()*100); rr['lock_hit_pct']=float((base.reason=='LOCK_STOP').mean()*100)
json.dump(rr,open(os.path.join(OUT,'base_rr.json'),'w'),indent=1)
# funnel
f={}
f['bars_total']=int(len(F)); f['sweep_signals(buy/sell)']=len(sd)
f['sweep_with_signal']=int((sd.sig.isin(['BUY','SELL'])).sum())
cfg0=dict(sim.BASE)
def cnt(**kw):
    c=dict(cfg0); c.update(kw)
    return int(sum(sim.passes(r,c) for r in C))
f['pass_all_model_gates(should_trade)']=cnt()
f['pass if conf gate off']=cnt(thr=0)
f['pass if conf+rsi off']=cnt(thr=0,rsi_buy=999,rsi_sell=-1)
f['pass if conf+rsi+ema off']=cnt(thr=0,rsi_buy=999,rsi_sell=-1,ema_block=False)
f['pass if conf+rsi+ema+agree off']=cnt(thr=0,rsi_buy=999,rsi_sell=-1,ema_block=False,min_agree=0)
f['final conf>=0.64 only']=int((sd.conf>=.64).sum()); f['sweep_conf>=0.64 only']=int((sd.sweep_conf>=.64).sum())
f['rsi blocks (of conf>=.64)']=int(((sd.conf>=.64)&(((sd.sig=='BUY')&(sd.rsi>70))|((sd.sig=='SELL')&(sd.rsi<30)))).sum())
f['trades taken in sim']=len(base)
json.dump(f,open(os.path.join(OUT,'funnel.json'),'w'),indent=1); print('FUNNEL',f,flush=True)

# ablations
def run(name,cfg):
    d,oe=sim.simulate(C,F,cfg); d=tag(d) if len(d) else d
    row=dict(name=name,n=len(d))
    if len(d)==0: return row,d
    for col,nm in (('pnl_bot','bot'),('pnl_real','real'),('pnl_gross','gross')):
        m=sim.metrics(d,col); row[f'net_{nm}']=round(m['net'],0)
        if nm=='real': row.update(wr=round(m['wr'],1),pf=round(m['pf'],2),maxdd=round(m['maxdd'],0),avg_w=round(m['avg_w'],0),avg_l=round(m['avg_l'],0))
    for lab,sub in (('y2024',d[d.year==2024]),('y2025',d[d.year==2025]),('y2026',d[d.year==2026])):
        row[f'real_{lab}']=round(float(sub.pnl_real.sum()),0); row[f'n_{lab}']=len(sub)
    return row,d
A=[('BASE (live config)',{}),
 ('RSI block OFF',dict(rsi_buy=999,rsi_sell=-1)),('RSI block 78/22',dict(rsi_buy=78,rsi_sell=22)),
 ('Conf gate 0.60',dict(thr=.60)),('Conf gate 0.55',dict(thr=.55)),('Conf gate 0.50',dict(thr=.50)),('Conf gate 0.40',dict(thr=.40)),('Conf gate OFF',dict(thr=0)),('Conf gate 0.70',dict(thr=.70)),('Conf gate 0.75',dict(thr=.75)),
 ('Gate on raw sweep conf>=0.64 (no model adjust)',dict(use_raw_sweep_conf=True)),
 ('EMA-oppose block OFF',dict(ema_block=False)),('Min models agree 2',dict(min_agree=2)),('Min models agree 4',dict(min_agree=4)),
 ('Tokyo block OFF',dict(tokyo_min=0)),('Bad UTC hours filter OFF',dict(bad_hours=())),
 ('Profit lock OFF',dict(lock=False)),('Profit lock activate 150 keep 50',dict(lock_act=150,lock_keep=50)),('Profit lock activate 60 keep 20',dict(lock_act=60,lock_keep=20)),
 ('Cooldown OFF (win0/loss0, no loss pause)',dict(cd_win=0,cd_loss=0,loss_cd_min=0,consec_n=99)),('Cooldown 3 bars after loss',dict(cd_loss=3)),
 ('TP = 1.0R (RR 1:1)',dict(tp_mode='r',tp_r=1.0)),('TP = 1.5R',dict(tp_mode='r',tp_r=1.5)),('TP = 2R',dict(tp_mode='r',tp_r=2.0)),('TP = 3R',dict(tp_mode='r',tp_r=3.0)),('TP = 0.8R',dict(tp_mode='r',tp_r=0.8)),
 ('SL x0.7 (tighter, TP kept)',dict(sl_scale=.7)),('SL x1.3 (wider, TP kept)',dict(sl_scale=1.3)),
 ('Size 1 contract always',dict(size_mode=1)),('Max 1 position at a time',dict(max_pos=1)),('Max 2 positions',dict(max_pos=2)),
 ('Session loss limits OFF',dict(session_limits=False)),
 ('Longs only',dict(long_only=True)),('Shorts only',dict(short_only=True)),('Skip Sunday',dict(skip_sunday=True)),('Day session (9:30-16:00 ET) only',dict(only_session='day')),('Overnight session only',dict(only_session='overnight')),
]
rows=[];store={}
for nm,cfg in A:
    r,d=run(nm,cfg); rows.append(r); store[nm]=cfg; print(r,flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT,'ablations.csv'),index=False)
ab=pd.DataFrame(rows)
# combos of top singles
top=ab[ab.name!='BASE (live config)'].sort_values('net_real',ascending=False).head(6)
print('TOP SINGLES',top[['name','n','net_real','pf']].to_string(),flush=True)
crows=[]
names=list(top.name)
for k in (2,3):
    for combo in itertools.combinations(names[:5],k):
        cfg={}; ok=True
        for nm in combo:
            for kk,vv in store[nm].items():
                if kk in cfg and cfg[kk]!=vv: ok=False
                cfg[kk]=vv
        if not ok: continue
        r,d=run(' + '.join(combo),cfg); crows.append(r); print(r,flush=True)
        pd.DataFrame(crows).to_csv(os.path.join(OUT,'ablation_combos.csv'),index=False)
print('DONE',flush=True)
