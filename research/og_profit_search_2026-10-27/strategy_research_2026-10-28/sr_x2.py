from sr_x import *
def load(tk):
    d=pd.read_csv(f'yf_{tk}_1h.csv',index_col=0,parse_dates=True); d.index=pd.to_datetime(d.index,utc=True).tz_convert(ET); return d[['Open','Close']].dropna()
ES=load('ES'); NQy=load('NQ')
def feat(Y,D,prevD):
    # prior ref = close of 15:00 ET hourly bar on prevD (16:00 ET); pre-open ref = open of 09:00 hourly bar on D (09:00 ET). both known before 09:30
    try:
        r=Y.loc[pd.Timestamp(f'{prevD} 15:00',tz=ET),'Close']; x=Y.loc[pd.Timestamp(f'{D} 09:00',tz=ET),'Open']
        return float(x/r-1)
    except KeyError: return np.nan
rows=[]
for k,D in enumerate(days):
    if k==0: continue
    pD=days[k-1]; a=sess[D]['a']
    i9=pos(D,'09:00',2)
    if i9 is None: continue
    m_mnq=o[i9]/sess[pD]['cl']-1
    rows.append(dict(d=D,mnq_pct=m_mnq,es_pct=feat(ES,D,pD),nqy_pct=feat(NQy,D,pD)))
F=pd.DataFrame(rows).set_index('d')
ok=F.dropna(); print('days',len(F),'with ES+NQ(yf)',len(ok))
print('ALIGNMENT: corr(MNQ local pre-open move, yf NQ=F move) = %.3f ; corr(MNQ, ES)=%.3f ; corr(NQ yf, ES)=%.3f'%(ok.mnq_pct.corr(ok.nqy_pct),ok.mnq_pct.corr(ok.es_pct),ok.nqy_pct.corr(ok.es_pct)))
print('median |MNQ-yfNQ| diff (bp):',round(float((ok.mnq_pct-ok.nqy_pct).abs().median()*1e4),1),' 99th pct (bp):',round(float((ok.mnq_pct-ok.nqy_pct).abs().quantile(.99)*1e4),1))
F['div']=F.mnq_pct-F.es_pct   # NQ minus ES overnight (pre-open) relative move, pct
bad=(F.mnq_pct-F.nqy_pct).abs()>0.004; print('days where yf NQ disagrees with local MNQ by >40bp (possible roll/stale):',int(bad.sum()))
F.to_csv('x2_features.csv')
def filt(name,r):
    es=r['es_pct']; dv=r['div']; g=r['g']
    if np.isnan(es): return None
    sg=np.sign(g)
    if name=='F1_ES_agrees': return np.sign(es)==sg
    if name=='F2_ES_disagrees': return np.sign(es)!=sg
    if name=='F3_big_div': return abs(dv)>=MED
    if name=='F4_small_div': return abs(dv)<MED
    if name=='F5_NQ_led(div same sign as gap)': return np.sign(dv)==sg
    if name=='F6_NQ_lagged(div opp sign)': return np.sign(dv)!=sg
    if name=='F7_ES_move_>=half_NQ_move_same_dir': return np.sign(es)==sg and abs(es)>=0.5*abs(r['mnq_pct'])
MED=float(F['div'].abs().median()); print('median |div| bp',round(MED*1e4,1))
FILTERS=['F1_ES_agrees','F2_ES_disagrees','F3_big_div','F4_small_div','F5_NQ_led(div same sign as gap)','F6_NQ_lagged(div opp sign)','F7_ES_move_>=half_NQ_move_same_dir']
CELLS=[(0.75,90),(0.70,120),(0.80,120),(0.65,90)]
res=[]
months=lambda df:((df.entry_time.max()-df.entry_time.min()).days/30.4)
for q,H in CELLS:
    rows_=ny2(q,H,stop=300)
    for r in rows_: r.update(F.loc[r['d']].to_dict() if r['d'] in F.index else dict(es_pct=np.nan,div=np.nan,mnq_pct=np.nan))
    allr=dfm(rows_); 
    for fn in ['NONE']+FILTERS:
        def sel(rs): return [r for r in rs if (fn=='NONE') or bool(filt(fn,r))]
        kept=sel(rows_); kept_df=dfm(kept) if kept else pd.DataFrame()
        if len(kept_df)<3: continue
        dropped=[r for r in rows_ if r not in kept]
        s=st(kept_df); kA=kept_df[kept_df.d<A1]; kB=kept_df[kept_df.d>=B0]; sA=st(kA) if len(kA)>=3 else {}; sB=st(kB) if len(kB)>=3 else {}
        # trades per month over data months
        tpm=len(kept_df)/22.6; tpmB=len(kB)/10.7
        # paired: kept - unfiltered per day delta = -(dropped pnl)
        dd=-pd.Series({r['d']:r['pnl'] for r in dropped}) if dropped else pd.Series(dtype=float)
        dd.index=[pd.Timestamp(x).date() for x in dd.index]
        ddB=dd[[x>=pd.Timestamp(B0).date() for x in dd.index]] if len(dd) else dd
        # slip/delay re-run
        def rr(**kw):
            rs=ny2(q,H,stop=300,**kw)
            for r in rs: r.update(F.loc[r['d']].to_dict() if r['d'] in F.index else dict(es_pct=np.nan,div=np.nan,mnq_pct=np.nan))
            k=[r for r in rs if (fn=='NONE') or bool(filt(fn,r))]; return st(dfm(k))['net'] if len(k)>=3 else None
        res.append(dict(q=q,H=H,filter=fn,n=len(kept_df),tpm=round(tpm,1),tpmB=round(tpmB,1),net=s['net'],pf=s['pf'],dd=s['dd'],corr=cb(kept_df),netA=sA.get('net'),nA=len(kA),netB=sB.get('net'),nB=len(kB),pfB=sB.get('pf'),ddB=sB.get('dd'),
            folds=s['folds_pos'],boot_p=s['boot_p'],dropped_n=len(dropped),dropped_pnl=round(sum(r['pnl'] for r in dropped)),pboot_filter_better_all=pboot(dd.values) if len(dd) else None,pboot_filter_better_B=pboot(ddB.values) if len(ddB) else None,
            slip4=rr(tk=4),delay3=rr(dly=3)))
    print('cell',q,H,flush=True)
R=pd.DataFrame(res); R.to_csv('x2_all.csv',index=False); pd.set_option('display.width',320); pd.set_option('display.max_columns',40)
print('variants',len(R))
for q,H in CELLS: print(R[(R.q==q)&(R.H==H)].drop(columns=['q','H']).to_string(index=False))