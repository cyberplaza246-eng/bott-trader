import os, sys, json, itertools
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import multiprocessing as mp
H=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,H)
import rv_engine as E
if __name__=='__main__':
    phase=sys.argv[1]; nproc=int(sys.argv[2])
    specs=json.load(open(os.path.join(H,f'specs_{phase}.json')))
    outp=os.path.join(H,'out',f'results_{phase}.jsonl')
    done=set()
    if os.path.exists(outp):
        for l in open(outp):
            try: done.add(json.loads(l)['name'])
            except Exception: pass
    items=[(n,{k:(tuple(v) if isinstance(v,list) else v) for k,v in s.items()}) for n,s in specs.items() if n not in done]
    print(len(items),'to run',flush=True)
    with mp.Pool(nproc) as pool, open(outp,'a',buffering=1) as f:
        for r in pool.imap_unordered(E.job,items):
            f.write(json.dumps(r,default=str)+'\n'); print(r['name'],r.get('res',{}).get('all'),r.get('err','')[:200],flush=True)
