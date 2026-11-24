from sr_x import *
SCH={
 'S0_flat1':lambda r:1,'S1_flat2':lambda r:2,
 'S2_2x_if_q90':lambda r:2 if r['absg']>=r['t90'] else 1,
 'S3_2x_if_400pt':lambda r:2 if r['absg']>=400 else 1,
 'S4_2x_if_500pt':lambda r:2 if r['absg']>=500 else 1,
 'S5_bucket_q85_q95_1-2-3':lambda r:3 if r['absg']>=r['t95'] else (2 if r['absg']>=r['t85'] else 1),
 'S6_bucket_300_450pt_1-2-3':lambda r:3 if r['absg']>=450 else (2 if r['absg']>=300 else 1),
 'S7_ANTI_2x_if_below_q90':lambda r:1 if r['absg']>=r['t90'] else 2}
CELLS=[(0.75,90),(0.70,120),(0.80,120),(0.65,90)]
def scaled(rows,f):
    out=[]
    for r in rows:
        k=f(r); x=dict(r); x['pnl']=r['pnl']*k; x['k']=k; out.append(x)
    return pd.DataFrame(out)
def run(q,H,fn,stop=300,**kw):
    rows=ny2(q,H,stop=stop,**kw); return scaled(rows,fn)
if __name__=='__main__':
    res=[];nv=0
    for (q,H) in CELLS:
        base1=dfm(run(q,H,SCH['S0_flat1']).to_dict('records')) if False else run(q,H,SCH['S0_flat1'])
        for nm,fn in SCH.items():
            nv+=1
            d=run(q,H,fn); dfA=d[(d.d>=A0)&(d.d<A1)]; dfB=d[d.d>=B0]
            s=st(dfm(d.to_dict('records'))); sB=st(dfm(dfB.to_dict('records'))); sA=st(dfm(dfA.to_dict('records')))
            # paired vs flat1 per-day delta
            d1=run(q,H,SCH['S0_flat1']); dd=dayser(d).sub(dayser(d1),fill_value=0)
            ddB=dd[dd.index>=pd.Timestamp(B0).date()]
            w=worst(d); wB=worst(dfB)
            s4=st(dfm(run(q,H,fn,tk=4).to_dict('records')))['net']; d3=st(dfm(run(q,H,fn,dly=3).to_dict('records')))['net']; d3s4=st(dfm(run(q,H,fn,dly=3,tk=4).to_dict('records')))['net']
            res.append(dict(q=q,H=H,scheme=nm,n=s['n'],n2plus=int((d.k>=2).sum()),net=s['net'],pf=s['pf'],dd=s['dd'],worst_trade=w['worst_trade'],worst_day=w['worst_day'],worst_week=w['worst_week'],train=s['train'],y26=s['y26'],folds=s['folds_pos'],boot_p=s['boot_p'],
                netA=sA.get('net'),netB=sB.get('net'),ddB=sB.get('dd'),worstB_day=wB['worst_day'],pf_B=sB.get('pf'),dB_vs_flat1=round(float(ddB.sum())),pboot_B_vs_flat1=pboot(ddB.values),pboot_all_vs_flat1=pboot(dd.values),slip4=s4,delay3=d3,delay3_slip4=d3s4,maxk=int(d.k.max())))
        print('cell',q,H,'done',flush=True)
    R=pd.DataFrame(res); R.to_csv('scale_all.csv',index=False); pd.set_option('display.width',300); pd.set_option('display.max_columns',40)
    print('variants',nv)
    for (q,H) in CELLS:
        print(R[(R.q==q)&(R.H==H)][['scheme','n','n2plus','net','pf','dd','worst_trade','worst_day','netA','netB','ddB','dB_vs_flat1','pboot_B_vs_flat1','pboot_all_vs_flat1','folds','slip4','delay3','delay3_slip4']].to_string(index=False))