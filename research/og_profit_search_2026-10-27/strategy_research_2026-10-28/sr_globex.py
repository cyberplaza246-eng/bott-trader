"""Globex 18:00 ET reopen deep dive (BACKTEST ONLY). weekday(Mon-Thu) vs Sunday; fade & continuation; thresholds; holds; delay; ORB; gap-fill reversion."""
import sys,os,json
from sr_gap2 import *
import sr_pub as P
ROLL=['2024-12-16','2025-03-17','2025-03-18','2025-06-16','2025-06-17','2025-09-16','2025-12-16','2026-03-16','2026-09-14','2026-09-15']
rolld=set()
for r in ROLL:
    for k in range(-4,2): rolld.add((pd.Timestamp(r)+pd.Timedelta(days=k)).strftime('%Y-%m-%d'))
SETS={'weekday':ev_globex(False),'sunday':ev_globex(True)}
def gtrades(ev,q=None,absp=None,H=60,mode='fade',tk=1,dly=0,exroll=False,perm=None):
    rows=[];past=[]
    for e_ in ev:
        mv=e_['mv']; a=e_['i']
        thr_ok=False
        if absp is not None: thr_ok=abs(mv)>=absp
        else: thr_ok=len(past)>=20 and abs(mv)>=(np.quantile(past,q) if q>0 else 0) if q>0 else len(past)>=20 or True
        past.append(abs(mv))
        if not thr_ok or mv==0: continue
        if exroll and e_['ds'] in rolld: continue
        side=-int(np.sign(mv)); side=side if mode=='fade' else -side
        i_in=a+dly; ex=a+H
        if not ok_dt(a,ex,H) or ex<=i_in+1: continue
        ent=o[i_in]+side*0.25*tk
        rows.append(mk(i_in,ent,ex,o[ex]-side*0.25*tk,side))
    return rows
def orb(ev,R,H,mode,tk=1,W=60,exroll=False):
    rows=[]
    for e_ in ev:
        a=e_['i']
        if exroll and e_['ds'] in rolld: continue
        if not ok_dt(a,a+R+W+H+2,R+W+H+2): continue
        rh=h[a:a+R].max(); rl=l[a:a+R].min()
        for i in range(a+R,a+R+W):
            s_=1 if c[i]>rh else (-1 if c[i]<rl else 0)
            if s_==0: continue
            side=s_ if mode=='brk' else -s_
            i_in=i+1; ex=i_in+H; ent=o[i_in]+side*0.25*tk
            rows.append(mk(i_in,ent,ex,o[ex]-side*0.25*tk,side)); break
    return rows
def gapfill(ev,q,tk=1,Hmax=180,exroll=False):
    rows=[];past=[]
    for e_ in ev:
        mv=e_['mv']; a=e_['i']; ok=len(past)>=20 and abs(mv)>=np.quantile(past,q); past.append(abs(mv))
        if not ok or mv==0 or (exroll and e_['ds'] in rolld): continue
        side=-int(np.sign(mv)); ent=o[a]+side*0.25*tk; tgt=ent-side*abs(mv)+0*side; stop=ent+side*(-1)*(-abs(mv))  # stop = 1x gap beyond entry (against)
        stop=ent-side*abs(mv)*1.0; stop=ent+(-side)*abs(mv)  # against direction
        ex=a+Hmax
        if not ok_dt(a,ex,Hmax): continue
        j,px,_=walk(a,ex,side,stop,o[a]-side*abs(mv)*1.0+0,tk)
        rows.append(mk(a,ent,j,px,side))
    return rows
EXR=['all','exroll']
def spec_list():
    S=[]
    for sn in SETS:
        for thr in [('q',0.0),('q',0.5),('q',0.75),('q',0.9),('abs',20),('abs',40)]:
            for H in (15,30,60,90,180,360):
                for mode in ('fade','cont'): S.append(('gap',sn,thr,H,mode))
        for R in (15,30):
            for H in (60,120,240):
                for mode in ('brk','rev'): S.append(('orb',sn,R,H,mode))
        for q in (0.5,0.75): S.append(('fill',sn,q,180,'fade'))
    return S
def build(sp,tk=1,dly=0,exroll=False):
    k,sn=sp[0],sp[1]; ev=SETS[sn]
    if k=='gap':
        thr,H,mode=sp[2],sp[3],sp[4]
        return gtrades(ev,q=thr[1] if thr[0]=='q' else None,absp=thr[1] if thr[0]=='abs' else None,H=H,mode=mode,tk=tk,dly=dly,exroll=exroll)
    if k=='orb': return orb(ev,sp[2],sp[3],sp[4],tk=tk,exroll=exroll)
    return gapfill(ev,sp[2],tk=tk,exroll=exroll)
def nm(sp):
    if sp[0]=='gap': return f"gap|{sp[1]}|{sp[2][0]}{sp[2][1]}|H{sp[3]}|{sp[4]}"
    if sp[0]=='orb': return f"orb|{sp[1]}|R{sp[2]}|H{sp[3]}|{sp[4]}"
    return f"fill|{sp[1]}|q{sp[2]}"
def cb(df):
    a=df.groupby(df.exit_time.dt.tz_convert(ET).dt.date).pnl.sum(); b=P.base.groupby(P.base.exit_time.dt.tz_convert(ET).dt.date).pnl.sum()
    j=pd.concat([a,b],axis=1).fillna(0); return round(float(j.corr().iloc[0,1]),3)
if __name__=='__main__':
    print('events',{k:len(v) for k,v in SETS.items()},flush=True)
    rows=[]
    for sp in spec_list():
        df=dfm(build(sp)); s=st(df) if len(df)>=3 else dict(n=len(df))
        s['name']=nm(sp); s['set']=sp[1]; s['kind']=sp[0]; s['sp']=repr(sp); rows.append(s)
    Rr=pd.DataFrame(rows); Rr.to_csv('globex_all.csv',index=False); print('variants',len(Rr))
    Rr['fp']=Rr.folds_pos.str.split('/').str[0].astype(float)
    print(Rr.groupby(['set','kind']).agg(n=('net','size'),mean_net=('net','mean'),pos=('net',lambda x:(x>0).sum()),best=('net','max')).round(0))
    g=Rr[Rr.kind=='gap']; g['mode']=g.name.str.split('|').str[-1]
    print(g.groupby(['set','mode']).agg(mean_net=('net','mean'),pos=('net',lambda x:(x>0).sum()),best=('net','max')).round(0))
    rob=Rr[(Rr.train>0)&(Rr.y26>0)&(Rr.fp>=4)&(Rr.boot_p>=0.9)]
    print('screen pass',len(rob)); 
    out=[]
    for _,r in rob.iterrows():
        sp=eval(r.sp); rec=dict(name=r['name'],n=r.n,net=r.net,pf=r.pf,dd=r.dd,train=r.train,y26=r.y26,folds=r.folds_pos,boot_p=r.boot_p,t=r.t)
        for lab,tk,dl,er in (('slip4',4,0,False),('delay1',1,1,False),('delay3',1,3,False),('exroll',1,0,True)):
            if sp[0]!='gap' and dl>0: rec[lab]='na'; continue
            rec[lab]=st(dfm(build(sp,tk,dl,er))).get('net')
        rec['corr']=cb(dfm(build(sp))); out.append(rec); print(rec)
    pd.DataFrame(out).to_csv('globex_robust.csv',index=False)
    print(Rr.sort_values('net',ascending=False).head(12)[['name','n','net','pf','dd','t','train','y26','folds_pos','boot_p']].to_string())