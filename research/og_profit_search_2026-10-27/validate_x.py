import sys,os
H=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,H)
import validate_run as VR
_orig=VR.specs
def specs():
    S=_orig(); X={}
    for tag in ('A','D'):
        sp=S[tag]
        X[f's_{tag}_tpthru2']=dict(sp,tp_through=0.5)
        X[f's_{tag}_tpthru4']=dict(sp,tp_through=1.0)
        X[f's_{tag}_tpthru2_slip2']=dict(sp,tp_through=0.5,slip=0.5)
        X[f'x_{tag}_early']=dict(sp,start='2024-09-25')
    return X
VR.specs=specs
if __name__=='__main__':
    import multiprocessing as mp, json, time
    n=int(sys.argv[1]); S=specs(); outp=os.path.join(VR.V,'results_runs.jsonl')
    with mp.Pool(n) as pool, open(outp,'a',buffering=1) as f:
        for r in pool.imap_unordered(VR.job,list(S.items())):
            f.write(json.dumps(r,default=str)+'\n'); print(r['name'],r.get('res',{}).get('all'),r.get('err','')[:300],flush=True)
