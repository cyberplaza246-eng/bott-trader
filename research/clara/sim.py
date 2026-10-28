import pandas as pd,numpy as np,itertools,json
b=pd.read_pickle('b2.pkl')
FOMC={'2024-09-18','2024-11-07','2024-12-18','2025-01-29','2025-03-19','2025-05-07','2025-06-18','2025-07-30','2025-09-17','2025-10-29','2025-12-10','2026-01-28','2026-03-18','2026-04-29','2026-06-17','2026-07-29','2026-09-16'}
HALF={'2024-11-29','2024-12-24','2025-07-03','2025-11-28','2025-12-24','2026-11-27'}
AFTER={'2024-12-02','2024-12-26','2025-07-07','2025-12-01','2025-12-26'}
def skipday(d):
    s=str(d.date())
    if s in FOMC or s in HALF or s in AFTER: return True
    return d.month in (3,6,9,12) and 8<=d.day<=15  # rollover week approx (guess)
b['d']=b.index.normalize()
days={}
for d,g in b.groupby('d'):
    g=g.between_time('09:32','15:30')
    if len(g)>150 and d.weekday()<5: days[d]=g
def run(P,days=days):
    out=[]
    for d,g in days.items():
        if skipday(d): continue
        o=g[g.index<=d+pd.Timedelta(minutes=590)]
        H,L=o.high.max(),o.low.min();W=H-L
        if W>P['wmax']: continue
        tp=(g.high+g.low+g.close)/3;vw=((tp*g.volume).cumsum()/g.volume.cumsum()).values
        C,Hh,Ll,Op=g.close.values,g.high.values,g.low.values,g.open.values;idx=g.index
        start=len(o);i=start;n=0
        cut=np.searchsorted(idx,d+pd.Timedelta(hours=P['cut']))
        while i<len(g)-1 and n<2:
            s=0
            if i<cut:
                if C[i]>H+P['buf'] and C[i]>vw[i]: s=1
                elif C[i]<L-P['buf'] and C[i]<vw[i]: s=-1
            if s==0: i+=1;continue
            e=Op[i+1];stop=P['k']*W
            if stop>200 or stop<P['smin']: i+=1;continue
            sp=e-s*stop;best=e;locked=False;j=i+1;ex=None
            while j<len(g):
                # intrabar stop check
                if (s==1 and Ll[j]<=sp) or (s==-1 and Hh[j]>=sp):
                    px=sp if (s==1 and Op[j]>sp) or (s==-1 and Op[j]<sp) else Op[j]
                    ex=(j,px,'stop');break
                if idx[j]>=d+pd.Timedelta(minutes=930): ex=(j,C[j],'close');break
                best=max(best,C[j]) if s==1 else min(best,C[j])
                mfe=s*(best-e)
                if mfe>=P['lockR']*stop:
                    locked=True
                    nsp=e+s*max(P['lockf']*mfe, mfe-P['trail']*stop)
                    sp=max(sp,nsp) if s==1 else min(sp,nsp)
                j+=1
            if ex is None: ex=(len(g)-1,C[-1],'close')
            pts=s*(ex[1]-e)-0.5
            out.append(dict(et=idx[i],s=s,e=e,x=ex[1],why=ex[2],pts=pts,usd=pts*2-4))
            n+=1;i=ex[0]+1
            if pts>0: break
    return pd.DataFrame(out)
def stats(t):
    if len(t)==0: return {}
    eq=t.usd.cumsum();dd=(eq.cummax().clip(lower=0)-eq).max()
    return dict(n=len(t),wr=round((t.usd>0).mean(),3),net=round(t.usd.sum()),per=round(t.usd.mean(),1),dd=round(dd),aw=round(t.usd[t.usd>0].mean()),al=round(t.usd[t.usd<=0].mean()))
if __name__=='__main__':
    tr={d:g for d,g in days.items() if d<pd.Timestamp('2026-01-01')}
    res=[]
    for wmax,k,lockR,lockf,trail,cut,buf in itertools.product([200,250,400],[0.8,1.0,1.2],[0.5,0.8,1.2],[0.3,0.5],[0.6,1.0,9],[11,12,14],[0,5]):
        P=dict(wmax=wmax,k=k,lockR=lockR,lockf=lockf,trail=trail,cut=cut,buf=buf,smin=20)
        st=stats(run(P,tr));st.update(P);res.append(st)
    r=pd.DataFrame(res);r['score']=r.net-r.dd
    r.sort_values('score',ascending=False).to_csv('grid.csv',index=False)
    print(r.sort_values('score',ascending=False).head(10).to_string())
