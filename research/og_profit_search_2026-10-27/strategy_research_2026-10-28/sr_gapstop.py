import sys,io,json,pickle,contextlib
src=open('sr_gap2.py',encoding='utf-8-sig').read().split('# ---- run ----')[0]
with contextlib.redirect_stdout(io.StringIO()): exec(src,globals())
def gf_stop(stop_pts,dly,tk=1,q=.75,Hm=90):
    rows=[];past=[]
    for e_ in EV['ny']:
        mv=e_['mv'];a=e_['i']
        ok=len(past)>=20 and abs(mv)>=np.quantile(past,q); past.append(abs(mv))
        if not ok or mv==0: continue
        side=-int(np.sign(mv)); i_in=a+dly; ex=a+Hm
        if not ok_dt(a,ex,Hm): continue
        ent=o[i_in]+side*0.25*tk
        if stop_pts is None: rows.append(mk(i_in,ent,ex,o[ex]-side*0.25*tk,side)); continue
        stop=ent-side*stop_pts
        j,px,why=walk(i_in,ex,side,stop,None,tk)
        r=mk(i_in,ent,j,px,side); r['why']=why; rows.append(r)
    return rows
out={}
for dly in(0,1,5):
  for sp in(None,60,80,100,125,150,200,250):
    r=dfm(gf_stop(sp,dly)); s=st(r)
    nst=int((pd.DataFrame(gf_stop(sp,dly)).get('why',pd.Series()).eq('stop')).sum()) if sp else 0
    out[f'd{dly}_s{sp}']=dict(n=len(r),net=round(r.pnl.sum()),pf=s.get('pf'),dd=s.get('dd'),train=s.get('train'),y26=s.get('y26'),folds=s.get('folds_pos'),boot_p=s.get('boot_p'),worst=round(r.pnl.min()),stops=nst)
    print(f'd{dly} s{sp}',out[f'd{dly}_s{sp}'],flush=True)
json.dump(out,open('gapstop.json','w'),indent=1)
# netting: opposite-direction baseline position open at entry
r=dfm(gf_stop(None,0))
b=base.copy(); b['entry_time']=pd.to_datetime(b.entry_time,utc=True); b['exit_time']=pd.to_datetime(b.exit_time,utc=True)
n_opp=0;n_same=0;pnl_opp=0;pnl_same=0;rows=[]
for _,t in r.iterrows():
    op=b[(b.entry_time<=t.entry_time)&(b.exit_time>t.entry_time)]
    d=op.dir.sum() if len(op) else 0
    opp=(d*t.dir<0)
    if opp: n_opp+=1;pnl_opp+=t.pnl
    elif len(op): n_same+=1;pnl_same+=t.pnl
print('opp-dir baseline open at entry: n',n_opp,'pnl',round(pnl_opp),'| same-dir open n',n_same,'pnl',round(pnl_same),'| total n',len(r))
# baseline positions open count at gap entry (cap 3)
cnt=[len(b[(b.entry_time<=t.entry_time)&(b.exit_time>t.entry_time)]) for _,t in r.iterrows()]
print('baseline open count at entry dist',pd.Series(cnt).value_counts().to_dict())
