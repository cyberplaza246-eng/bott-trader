"""Public-idea sleeves on MNQ 1m (BACKTEST ONLY). Costs: $4 RT + slip ticks on market/stop/time exits; limit fills/targets need price to trade THROUGH by 1 tick. No lookahead."""
import os,sys,json
os.environ['PYTHONDONTWRITEBYTECODE']='1'; sys.dont_write_bytecode=True
R=os.path.dirname(os.path.abspath(__file__))
import warnings; warnings.filterwarnings('ignore')
import io,contextlib
with contextlib.redirect_stdout(io.StringIO()): import sr_engine as S
E=S.E; import numpy as np, pandas as pd
ET=S.ET; d=E.load(); m1=d['m1']
ix=m1.index; o=m1.open.values; h=m1.high.values; l=m1.low.values; c=m1.close.values; v=m1.volume.values
etx=ix.tz_convert(ET); DS=np.array(etx.strftime('%Y-%m-%d')); MOD=(etx.hour*60+etx.minute).values
def T(ds,hm): return pd.Timestamp(f'{ds} {hm}',tz=ET).tz_convert('UTC')
def pos(ds,hm,tol=3):
    t=T(ds,hm); i=ix.searchsorted(t)
    return i if i<len(ix) and (ix[i]-t)<=pd.Timedelta(minutes=tol) else None
# sessions: NY 09:30-15:59
sess={}
for ds in pd.unique(DS):
    a=pos(ds,'09:30'); b=pos(ds,'15:59')
    if a is None or b is None or b-a<300: continue
    sess[ds]=dict(a=a,b=b,hi=h[a:b+1].max(),lo=l[a:b+1].min(),op=o[a],cl=c[b])
days=sorted(sess); 
def walk(i0,i1,side,stop,target,tk,first_no_tgt=False):
    """walk bars i0..i1-1 ; time exit at open of i1. returns exit_idx,px,reason"""
    sl=0.25*tk
    for i in range(i0,i1):
        if side>0:
            if l[i]<=stop: return i,min(o[i] if i>i0 or not first_no_tgt else stop,stop)-sl,'stop'
            if target is not None and not(first_no_tgt and i==i0) and h[i]>=target+0.25: return i,target,'tgt'
        else:
            if h[i]>=stop: return i,max(o[i] if i>i0 or not first_no_tgt else stop,stop)+sl,'stop'
            if target is not None and not(first_no_tgt and i==i0) and l[i]<=target-0.25: return i,target,'tgt'
    return i1,o[i1]-side*sl,'time'
def mk(i_in,px_in,i_out,px_out,side,qty=1):
    return dict(entry_time=ix[i_in],exit_time=ix[i_out],dir=side,pnl=qty*(side*(px_out-px_in)*2-4.0))
# ---------- sleeves ----------
def dipbuy(tk=1,rangefilt=True,tgtR=0.1,stopR=0.8):
    rows=[]
    for k,D in enumerate(days):
        if rangefilt:
            if k<20: continue
            avg=np.mean([sess[x]['hi']-sess[x]['lo'] for x in days[k-20:k]])
        s=sess[D]; Rg=s['hi']-s['lo']
        if rangefilt and Rg<avg: continue
        dt=pd.Timestamp(D); sd=dt if dt.weekday()<4 else dt+pd.Timedelta(days=2); ed=sd+pd.Timedelta(days=1)
        sd=sd.strftime('%Y-%m-%d'); ed=ed.strftime('%Y-%m-%d')
        a=pos(sd,'18:00',10); z=pos(sd,'03:00',10) if False else pos(ed,'03:00'); e=pos(ed,'08:25')
        if a is None or z is None or e is None: continue
        if not(s['lo']<o[a]<s['hi']): continue
        lim=s['lo']; fill=None
        for i in range(a,z):
            if h[i]>s['hi']: break
            if l[i]<=lim-0.25: fill=i; break
        if fill is None: continue
        entry=lim+(0.25*(tk-1) if tk>1 else 0.0)   # stress: worse limit fill
        j,px,_=walk(fill,e,1,entry-stopR*Rg,entry+tgtR*Rg,tk,first_no_tgt=True)
        rows.append(mk(fill,entry,j,px,1))
    return rows
def morning(tk=1,trend=True,gap=True,until='11:30'):
    rows=[]
    for k,D in enumerate(days):
        if k<20: continue
        pc=sess[days[k-1]]['cl']; s=sess[D]
        if trend and not(pc>np.mean([sess[x]['cl'] for x in days[k-20:k]])): continue
        if gap and not(s['op']>pc): continue
        a=s['a']; r_end=pos(D,'10:30'); u=pos(D,until); e=pos(D,'15:55')
        if r_end is None or u is None or e is None: continue
        rh=h[a:r_end].max(); rl=l[a:r_end].min(); rw=rh-rl
        for i in range(r_end,u):
            if c[i]>rh:
                ent=o[i+1]+0.25*tk; j,px,_=walk(i+1,e,1,rl,c[i]+0.5*rw,tk)
                rows.append(mk(i+1,ent,j,px,1)); break
    return rows
def gapq(day_k,past):  # helper
    pass
def gapfade(q=0.75,H=90,confirm=False,tk=1,qty_fn=None):
    rows=[];past=[]
    for k,D in enumerate(days):
        if k==0: continue
        pc=sess[days[k-1]]['cl']; s=sess[D]; a=s['a']; g=s['op']-pc
        ok=len(past)>=20 and abs(g)>=(np.quantile(past,q) if q>0 else 0) and g!=0
        past.append(abs(g))
        if not ok: continue
        side=-int(np.sign(g)); qty=qty_fn(k) if qty_fn else 1
        if confirm:
            if not(side*(c[a+4]-o[a])>0): continue       # first 5m closes against the gap
            i_in=a+5; ent=o[i_in]+side*0.25*tk
        else: i_in=a; ent=o[a]+side*0.25*tk
        e=pos(D,'09:30')+H
        if e<=i_in+1: continue
        rows.append(mk(i_in,ent,e,o[e]-side*0.25*tk,side,qty))
    return rows
def gapfill_ce(tk=1,thr=0.0025):
    rows=[]
    for k,D in enumerate(days):
        if k==0: continue
        pc=sess[days[k-1]]['cl']; s=sess[D]; a=s['a']; g=s['op']-pc
        if abs(g)/pc<thr: continue
        side=-int(np.sign(g)); ent=o[a]+side*0.25*tk
        stop=ent-side*abs(g); tgt=pc
        j,px,_=walk(a,s['b'],side,stop,tgt,tk)
        rows.append(mk(a,ent,j,px,side))
    return rows
def ondrop(tk=1,z0=-2.0):
    rows=[];rets=[]
    for k,D in enumerate(days):
        if k==0: continue
        pc=sess[days[k-1]]['cl']; s=sess[D]; a=s['a']; r=s['op']/pc-1
        if len(rets)>=60:
            z=r/np.std(rets[-60:],ddof=1)
            if z<=z0 and c[a+4]>o[a]:
                e=a+30
                rows.append(mk(a+5,o[a+5]+0.25*tk,e,o[e]-0.25*tk,1))
        rets.append(r)
    return rows
# Asian-range breakout (Night Shift Quant), 5m bars from m1
m5=m1.resample('5min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
def asian(tk=1):
    ix5=m5.index; e5=ix5.tz_convert(ET); mod=(e5.hour*60+e5.minute).values; ds5=np.array(e5.strftime('%Y-%m-%d'))
    hh=m5.high.values;ll=m5.low.values;cc=m5.close.values;oo=m5.open.values;vv=m5.volume.values
    pc_=np.r_[np.nan,cc[:-1]]; tr=np.maximum(hh-ll,np.maximum(abs(hh-pc_),abs(ll-pc_))); atr=pd.Series(tr).rolling(14).mean().values
    med=pd.Series(atr).rolling(2880).median().shift(1).values; vavg=pd.Series(vv).rolling(20).mean().shift(1).values
    rows=[]
    for D in sorted(set(ds5)):
        dt=pd.Timestamp(D)
        if dt.weekday()>4: continue
        prev=(dt-pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        asian_idx=np.where(((ds5==prev)&(mod>=18*60))|((ds5==D)&(mod<2*60)))[0]
        if len(asian_idx)<60: continue
        rh=hh[asian_idx].max(); rl=ll[asian_idx].min()
        win=np.where((ds5==D)&(mod>=150)&(mod<240))[0]; n=0; busy=-1
        for i in win:
            if i<=busy or n>=2 or np.isnan(atr[i]) or np.isnan(med[i]): continue
            if atr[i]<=med[i] or vv[i]<1.4*vavg[i]: continue
            side=1 if cc[i]>rh else (-1 if cc[i]<rl else 0)
            if side==0: continue
            ent=oo[i+1]+side*0.25*tk; A=atr[i]; stop=ent-side*A; tgt=ent+side*2*A
            endi=np.where((ds5==D)&(mod==240))[0]
            if len(endi)==0: break
            e=endi[0]; sl=0.25*tk; exit_=None
            for j in range(i+1,e):
                if side>0:
                    if ll[j]<=stop: exit_=(j,min(oo[j],stop)-sl);break
                    if hh[j]>=tgt+0.25: exit_=(j,tgt);break
                else:
                    if hh[j]>=stop: exit_=(j,max(oo[j],stop)+sl);break
                    if ll[j]<=tgt-0.25: exit_=(j,tgt);break
            if exit_ is None: exit_=(e,oo[e]-side*sl)
            rows.append(dict(entry_time=ix5[i+1],exit_time=ix5[exit_[0]],dir=side,pnl=side*(exit_[1]-ent)*2-4.0)); n+=1; busy=exit_[0]
    return rows
# ---------- stats ----------
cut=pd.Timestamp('2026-01-01',tz='UTC')
FOLDS=[('2025-05-01','2025-08-01'),('2025-08-01','2025-11-01'),('2025-11-01','2026-02-01'),('2026-02-01','2026-05-01'),('2026-05-01','2026-08-01'),('2026-08-01','2026-10-01')]
def mdd(p): eq=np.cumsum(p); return float((np.maximum.accumulate(np.concatenate([[0],eq]))[1:]-eq).max()) if len(p) else 0.
rng=np.random.default_rng(7)
def dfm(rows): 
    df=pd.DataFrame(rows)
    if len(df)==0: return df
    return df.sort_values('exit_time').reset_index(drop=True)
def st(df):
    if len(df)<3: return dict(n=len(df))
    p=df.pnl.values; g=p[p>0].sum(); ls=-p[p<0].sum(); t=p.mean()/(p.std(ddof=1)/np.sqrt(len(p))) if p.std(ddof=1)>0 else 0
    fo=[df[(df.entry_time>=pd.Timestamp(a,tz='UTC'))&(df.entry_time<pd.Timestamp(b,tz='UTC'))].pnl.sum() for a,b in FOLDS]
    dd=df.groupby(df.exit_time.dt.tz_convert(ET).dt.date).pnl.sum().values
    bs=np.array([rng.choice(dd,len(dd)).sum() for _ in range(3000)])
    tr=df[df.entry_time<cut]; te=df[df.entry_time>=cut]
    return dict(n=len(p),net=round(p.sum()),pf=round(g/ls,2) if ls>0 else 99,win=round(100*(p>0).mean()),dd=round(mdd(p)),worst=round(p.min()),t=round(t,2),
        train=round(tr.pnl.sum()),n_tr=len(tr),y26=round(te.pnl.sum()),n_26=len(te),wf=round(sum(fo)),folds_pos=f'{sum(x>0 for x in fo)}/6',boot_p=round(float((bs>0).mean()),2))
base=pd.read_csv(os.path.join(R,'trades_base.csv'))
for cc_ in('entry_time','exit_time'): base[cc_]=pd.to_datetime(base[cc_],utc=True)
def comb(df): 
    x=pd.concat([base[['entry_time','exit_time','pnl']],df[['entry_time','exit_time','pnl']]]).sort_values('exit_time')
    s=st(x); return dict(net=s['net'],pf=s['pf'],dd=s['dd'],train=s['train'],y26=s['y26'],wf=s['wf'])
if __name__=='__main__':
    out={}
    specs={
     'dipbuy_filt':lambda tk:dipbuy(tk,True),'dipbuy_nofilt':lambda tk:dipbuy(tk,False),
     'morning_full':lambda tk:morning(tk),'morning_notrend':lambda tk:morning(tk,trend=False),'morning_nofilter':lambda tk:morning(tk,False,False),
     'gap_q75_H90':lambda tk:gapfade(0.75,90,False,tk),'gap_q75_H90_confirm5m':lambda tk:gapfade(0.75,90,True,tk),
     'gap_q50_H90_confirm5m':lambda tk:gapfade(0.5,90,True,tk),'gap_q0_H90_confirm5m':lambda tk:gapfade(0.0,90,True,tk),
     'gapfill_CE_0.25pct':lambda tk:gapfill_ce(tk),'ondrop_z2':lambda tk:ondrop(tk),'ondrop_z15':lambda tk:ondrop(tk,-1.5),'asian_breakout':lambda tk:asian(tk)}
    for nm,f in specs.items():
        r=dfm(f(1)); s=st(r); 
        if len(r)>=3:
            s['slip3_net']=st(dfm(f(3)))['net']; s['with_base']=comb(r); r.to_csv(os.path.join(R,f'pub_{nm}.csv'),index=False)
        out[nm]=s; print(nm,s,flush=True)
    json.dump(out,open(os.path.join(R,'pub_results.json'),'w'),indent=1,default=str)
