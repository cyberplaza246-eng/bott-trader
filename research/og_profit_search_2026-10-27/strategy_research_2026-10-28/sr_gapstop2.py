exec(open('sr_gapstop.py',encoding='utf-8-sig').read().split('out={}')[0])
for sp in(225,240,250,300,350,400,500):
    for dly in(0,1):
        r=dfm(gf_stop(sp,dly)); s=st(r)
        print(f'stop{sp} d{dly}',dict(net=round(r.pnl.sum()),pf=s['pf'],dd=s['dd'],train=s['train'],y26=s['y26'],folds=s['folds_pos'],boot_p=s['boot_p'],worst=round(r.pnl.min())),flush=True)
# policy simulation
b=base.copy(); b['entry_time']=pd.to_datetime(b.entry_time,utc=True); b['exit_time']=pd.to_datetime(b.exit_time,utc=True)
b=b.sort_values('entry_time').reset_index(drop=True)
def policy(sp,dly,cap=3):
    g=pd.DataFrame(gf_stop(sp,dly)); g['entry_time']=pd.to_datetime(g.entry_time,utc=True); g['exit_time']=pd.to_datetime(g.exit_time,utc=True)
    keep=[];blocked_b=set()
    skipped=0
    for _,t in g.iterrows():
        op=b[(b.entry_time<=t.entry_time)&(b.exit_time>t.entry_time)&(~b.index.isin(blocked_b))]
        if len(op) and (op.dir.sum()*t.dir<0): skipped+=1; continue
        if len(op)>=cap: skipped+=1; continue
        keep.append(t)
    k=pd.DataFrame(keep)
    # baseline trades blocked by gap position being open
    bl=[]
    ent_px=lambda tt: o[ix.get_loc(tt)]
    for _,t in k.iterrows():
        gi=ix.get_loc(t.entry_time); gp=o[gi]+t.dir*0.25
        w=b[(b.entry_time>t.entry_time)&(b.entry_time<t.exit_time)]
        for j,x in w.iterrows():
            p=o[ix.get_loc(x.entry_time)]
            prof=t.dir*(p-gp)>0
            if x.dir!=t.dir or not prof: bl.append(j)
    bl=set(bl)
    lost=b.loc[list(bl)].pnl.sum() if bl else 0
    return len(g),skipped,len(k),k.pnl.sum(),len(bl),lost
for sp,dly in((None,0),(None,1),(240,0),(240,1),(250,1)):
    n,sk,nk,gp,nb,lost=policy(sp,dly)
    print(f'POLICY stop{sp} d{dly}: gap trades {n} skipped(opp/cap) {sk} taken {nk} gap pnl {gp:.0f} | baseline trades blocked by open gap pos {nb} lost baseline pnl {lost:.0f} | NET effect {gp-lost:.0f}',flush=True)
