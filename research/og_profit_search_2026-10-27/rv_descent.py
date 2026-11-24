import os, sys, json, copy
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import multiprocessing as mp
H=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,H)
import rv_engine as E
COMP=E.COMPONENTS
def key(sp): return json.dumps(sp,sort_keys=True,default=list)
def neighbors(sp):
    out=[]
    for c in COMP:
        n=copy.deepcopy(sp); d=set(n.get('drop',[]))
        d^={c}; n['drop']=sorted(d); out.append(n)
    for lk in ['base','tight2','wide']:
        if sp.get('lock','base')!=lk: n=copy.deepcopy(sp); n['lock']=lk; out.append(n)
    for r in [2.0,2.5,3.0,3.5]:
        if sp.get('bo_r',2.0)!=r: n=copy.deepcopy(sp); n['bo_r']=r; out.append(n)
    for r in [1.0,1.5,2.0]:
        if sp.get('trend_r',1.5)!=r: n=copy.deepcopy(sp); n['trend_r']=r; out.append(n)
    n=copy.deepcopy(sp); n['revamp']=not sp.get('revamp',True); out.append(n)
    return out
def tup(sp): return {k:(tuple(v) if isinstance(v,list) else v) for k,v in sp.items()}
if __name__=='__main__':
    which=sys.argv[1]; nproc=int(sys.argv[2])   # which: train | y2026
    cache={}; outp=os.path.join(H,'out','results_descent.jsonl')
    if os.path.exists(outp):
        for l in open(outp):
            r=json.loads(l); 
            if 'spec_key' in r: cache[r['spec_key']]=r
    f=open(outp,'a',buffering=1)
    def score(r):
        m=r['res']['train' if which=='train' else 'y2026']
        return m['net']-0.75*m['dd']
    def evaluate(specs,pool):
        todo=[s for s in specs if key(s) not in cache]
        items=[(f'd{len(cache)+i}',tup(s)) for i,s in enumerate(todo)]
        for s,r in zip(todo,pool.imap(E.job,items)):
            r['spec_key']=key(s); cache[key(s)]=r; f.write(json.dumps(r,default=str)+'\n')
        return [cache[key(s)] for s in specs]
    with mp.Pool(nproc) as pool:
        cur={'drop':[]}; cr=evaluate([cur],pool)[0]; best=score(cr)
        log=open(os.path.join(H,'out',f'descent_{which}.log'),'w',buffering=1)
        print('start',which,round(best),file=log)
        for rnd in range(5):
            nb=neighbors(cur); rs=evaluate(nb,pool)
            cand=max(zip(rs,nb),key=lambda x:score(x[0])); sc=score(cand[0])
            print(f'round {rnd}: best neighbor score {sc:.0f} vs cur {best:.0f} spec {key(cand[1])} train={cand[0]["res"]["train"]["net"]} dd{cand[0]["res"]["train"]["dd"]} 2026={cand[0]["res"]["y2026"]["net"]} dd{cand[0]["res"]["y2026"]["dd"]}',file=log)
            if sc>best*1.02+50: cur=cand[1]; best=sc
            else: break
        print('FINAL',key(cur),file=log)
