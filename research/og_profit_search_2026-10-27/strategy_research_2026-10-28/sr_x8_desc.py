"""Addendum 8: afternoon / FOMC-day behaviour of the 3R baseline. BACKTEST ONLY. Part 1 = descriptive buckets from trades_base.csv."""
import pandas as pd, numpy as np, json, sys
ET='America/New_York'
FOMC=['2024-11-07','2024-12-18','2025-01-29','2025-03-19','2025-05-07','2025-06-18','2025-07-30','2025-09-17','2025-10-29','2025-12-10',
      '2026-01-28','2026-03-18','2026-04-29','2026-06-17','2026-07-29','2026-09-16']
def mdd(p):
    if len(p)==0: return 0.0
    eq=np.cumsum(p); return float((np.maximum.accumulate(np.concatenate([[0],eq]))[1:]-eq).max())
def met(df):
    if len(df)==0: return dict(n=0,net=0,pf=0,avg=0,dd=0,wr=0)
    p=df.sort_values('exit_time').pnl.values; g=p[p>0].sum(); l=-p[p<0].sum()
    return dict(n=len(p),net=round(float(p.sum())),pf=round(float(g/l),2) if l>0 else 99.0,avg=round(float(p.mean()),1),dd=round(mdd(p)),wr=round(100*float((p>0).mean())))
def load(path='trades_base.csv'):
    d=pd.read_csv(path,parse_dates=['entry_time','exit_time'])
    for c in ('entry_time','exit_time'): d[c]=pd.to_datetime(d[c],utc=True)
    e=d.entry_time.dt.tz_convert(ET); d['emin']=e.dt.hour*60+e.dt.minute; d['eday']=e.dt.strftime('%Y-%m-%d')
    d['fomc']=d.eday.isin(FOMC); d['win']=np.where(d.entry_time<pd.Timestamp('2026-01-01',tz='UTC'),'train','y2026')
    return d
def bucket(m):
    if m<9*60+30 or m>=18*60: return '0_overnight_pre-open (18:00-09:30)'
    if m<13*60+15: return '1_RTH 09:30-13:15'
    if m<14*60: return '2_13:15-14:00'
    if m<16*60: return '3_14:00-16:00'
    return '4_16:00-18:00'
if __name__=='__main__':
    d=load(); d['bucket']=d.emin.map(bucket)
    print('FOMC days in sample with any trade:',d[d.fomc].eday.nunique(),'of',len(FOMC),'; trades on FOMC dates',int(d.fomc.sum()))
    print(d.groupby('bucket').size())
    print(d.groupby(d.emin//60).agg(n=('pnl','size'),net=('pnl','sum'),avg=('pnl','mean')).round(1).T.to_string())
    rows=[]
    for fl,fn in ((False,'non-FOMC'),(True,'FOMC')):
        for w in ('all','train','y2026'):
            for b,g in d[(d.fomc==fl)&((d.win==w) if w!='all' else True)].groupby('bucket'):
                rows.append(dict(day=fn,win=w,bucket=b,**met(g)))
            rows.append(dict(day=fn,win=w,bucket='ALL',**met(d[(d.fomc==fl)&((d.win==w) if w!='all' else True)])))
    t=pd.DataFrame(rows); pd.set_option('display.width',250); print(t.to_string())
    t.to_csv('x8_buckets.csv',index=False)
    # distinct-day stats
    print('trading days per window FOMC/non:',d.groupby(['win','fomc']).eday.nunique().to_dict())
    # per FOMC day pnl
    f=d[d.fomc].groupby('eday').pnl.agg(['size','sum']).round(0); print(f.T.to_string())
    # trades open across 14:00 on FOMC day, entered before 13:15
    ex=d.exit_time.dt.tz_convert(ET)
    d['t1400']=pd.to_datetime(d.eday+' 14:00').dt.tz_localize(ET); d['t1600']=pd.to_datetime(d.eday+' 16:00').dt.tz_localize(ET)
    op=d[d.fomc&(d.entry_time.dt.tz_convert(ET)<d.t1400)&(ex>d.t1400)]
    print('FOMC trades open across 14:00:',met(op),'| by win',{w:met(g) for w,g in op.groupby('win')})
    # window P&L: hold-through ones exit before 16:00
    print(op[['eday','entry_time','exit_time','pnl','tag','reason']].assign(entry=lambda x:x.entry_time.dt.tz_convert(ET).dt.strftime('%H:%M'),exit=lambda x:x.exit_time.dt.tz_convert(ET).dt.strftime('%m-%d %H:%M')).drop(columns=['entry_time','exit_time']).to_string())
