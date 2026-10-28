import sys,pickle,numpy as np,pandas as pd
exec(open("/workspace/eval_passer/backtest_noreset.py").read().split("if __name__")[0])
d1=pd.read_parquet("/workspace/intrabar/1m_long.parquet"); TS=d1.index.as_unit("ns").asi8; C=d1.close.to_numpy()
X,Dd,Rg=pickle.load(open("/workspace/intrabar/diag.pkl","rb")); X=X[X.g!="6pm"]
V=pickle.load(open("/workspace/sixpm/streams.pkl","rb"))
DAYS[:]=[d for d in sorted(set(X.td)) if pd.Timestamp(d).weekday()<5 and d<="2026-10-07"]
def run2(name):
    rows=[]
    for si,s in enumerate(DAYS[:-25]):
        G=F.Flex3Guard(); hw=0.0; res=("open",0); ch=0
        for j,d in enumerate(DAYS[si:]):
            G.new_day(d); floor=min(hw-2000,100.0); G._floor=floor
            sim_day(G,EB.get(d,[]),fixed(1),None,False,None)
            if G.passed(): res=("pass",j+1); break
            if G.eval_pnl>=3000: ch=1
            if G.eval_pnl<=floor: res=("fail",j+1); break
            hw=max(hw,G.eval_pnl)
        rows.append(dict(start=s,eod=res[0],days=res[1],cons=ch))
    return pd.DataFrame(rows)
per=lambda t: np.asarray(t,str).astype("U4")
rows=[];EQ={}
for nm,six in V.items():
    x=pd.concat([X,six]); x["exit_ns"]=TS[np.minimum(x.kx.to_numpy(),len(TS)-1)]+ONE; x["stream"]=x.g.map({"v2":"v2","ONL":"addon","E1":"addon","6pm":"6pm"}); x["td"]=x.td.astype(str); x=x.sort_values(["td","k"])
    EB.clear(); EB.update(by_day(x))
    F.TARGET=1e12; G=F.Flex3Guard(); dl={}
    for d in DAYS: G.new_day(d); G._floor=-1e9; dl[d]=sim_day(G,EB.get(d,[]),fixed(1),None,False,None)[0]
    F.TARGET=3000.0; s=pd.Series(dl); EQ[nm]=s
    R=run2(nm); R["per"]=per(R.start)
    for pr,g in list(R.groupby("per"))+[("ALL",R)]:
        ss=s if pr=="ALL" else s[per(s.index)==pr]; e=ss.cumsum(); sx=six if pr=="ALL" else six[six.td.str[:4]==pr]
        rows.append(dict(variant=nm,period=pr,leg_net=round(sx.pnl.sum()),leg_n=len(sx),leg_win=round(100*(sx.pnl>0).mean(),1),
          pass_pct=round(100*(g.eod=="pass").mean(),1),bust_pct=round(100*(g.eod=="fail").mean(),1),med_days=g[g.eod=="pass"].days.median(),
          cons_hits=int(g.cons.sum()),bot_net=round(ss.sum()),worst_dd=round((e-e.cummax()).min())))
    print(nm,flush=True)
T=pd.DataFrame(rows); T.to_csv("/workspace/sixpm/results.csv",index=False); pickle.dump(EQ,open("/workspace/sixpm/eq.pkl","wb")); pd.set_option("display.width",250); print(T.to_string())
