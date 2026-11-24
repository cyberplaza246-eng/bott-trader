import sr_bigc as B, pandas as pd, numpy as np
P=B.P
for f,p,tf,mode,ex in [('range',dict(N=4),5,'cont','T60'),('news',dict(hm=600,R=1.5,nm='1000'),5,'cont','T60')]:
    idx,sd=B.signals(f,p,tf); df=P.dfm(B.run(idx,sd,tf,mode,ex).to_dict('records'))
    fo=[round(df[(df.entry_time>=pd.Timestamp(a,tz='UTC'))&(df.entry_time<pd.Timestamp(b,tz='UTC'))].pnl.sum()) for a,b in P.FOLDS]
    d=df.groupby(df.exit_time.dt.tz_convert(B.ET).dt.date).pnl.sum().sort_values(ascending=False)
    print(f,p,'folds',fo,'net',round(df.pnl.sum()),'ex top3 days',round(d.iloc[3:].sum()),'ex top10',round(d.iloc[10:].sum()),'top days',d.head(5).round().tolist())
    m=df.groupby(df.entry_time.dt.tz_convert(B.ET).dt.to_period('M')).pnl.sum(); print(' months green',int((m>0).sum()),'/',len(m))
    print(' RTH-only net', round(df[(df.entry_time.dt.tz_convert(B.ET).dt.hour*60+df.entry_time.dt.tz_convert(B.ET).dt.minute).between(570,959)].pnl.sum()))