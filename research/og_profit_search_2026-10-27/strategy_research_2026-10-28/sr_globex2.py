import sys; from sr_globex import *
reps=[('sunday',('q',0.5),360),('sunday',('abs',40),360),('sunday',('abs',40),180),('sunday',('q',0.0),180),('sunday',('abs',40),15),('weekday',('abs',20),90),('weekday',('q',0.9),90)]
def bt(sn,thr,H,tk=1,dly=0,mode='cont',exroll=False):
    return dfm(gtrades(SETS[sn],q=thr[1] if thr[0]=='q' else None,absp=thr[1] if thr[0]=='abs' else None,H=H,mode=mode,tk=tk,dly=dly,exroll=exroll))
for sn,thr,H in reps:
    df=bt(sn,thr,H); p=df.pnl.values; s=st(df)
    top=np.sort(p)[::-1]; d=df.groupby(df.exit_time.dt.tz_convert(ET).dt.date).pnl.sum()
    mons=(df.entry_time.max()-df.entry_time.min()).days/30.4
    dl={x:st(bt(sn,thr,H,dly=x))['net'] for x in (1,2,3,5,10)}
    c=P.comb(df)
    print(sn,thr,H,'| n',len(p),f'tpm {len(p)/mons:.1f}','net',s['net'],'pf',s['pf'],'dd',s['dd'],'worst',s['worst'],'| top3 share %.0f%%'%(100*top[:3].sum()/p.sum()),'ex-top5',round(p.sum()-top[:5].sum()),'| delays',dl,'| slip4',st(bt(sn,thr,H,tk=4))['net'],'d3+s4',st(bt(sn,thr,H,tk=4,dly=3))['net'],'| corr',cb(df),'| port net',c['net'],'dd',c['dd'],'wf',c['wf'],flush=True)
    # first print vs 1-min bar: avg first-bar move direction continuation
# lookahead / timing sanity: mean |gap| per set + first-minute continuation
for sn in SETS:
    ev=SETS[sn]; mv=np.array([e['mv'] for e in ev]); a=np.array([e['i'] for e in ev])
    r1=np.array([c[i]-o[i] for i in a]); r5=np.array([c[i+4]-o[i] for i in a])
    print(sn,'events',len(ev),'mean|gap|',round(np.abs(mv).mean(),1),'corr(gap, first-1m move) ',round(float(np.corrcoef(mv,r1)[0,1]),2),' corr(gap,0-5m move)',round(float(np.corrcoef(mv,r5)[0,1]),2),' share first1m same sign',round(float((np.sign(mv)==np.sign(r1)).mean()),2))