import sys,json,numpy as np
sys.argv=['x']
import sr_pub as P
days=P.days; sess=P.sess
rng=[sess[D]['hi']-sess[D]['lo'] for D in days]
def ratio(k):
    if k<25: return 1.0
    tr=np.mean(rng[k-20:k]); med=np.median(rng[:k]); return tr/med
out={}
for nm,fn in [('flat1',None),('hi_vol_x2',lambda k:2 if ratio(k)>=1 else 1),('lo_vol_x2',lambda k:2 if ratio(k)<1 else 1),('flat2',lambda k:2)]:
    r=P.dfm(P.gapfade(0.75,90,False,1,fn)); s=P.st(r); s['slip3']=P.st(P.dfm(P.gapfade(0.75,90,False,3,fn)))['net']; s['with_base']=P.comb(r); out[nm]=s; print(nm,s,flush=True)
json.dump(out,open('size_results.json','w'),indent=1,default=str)
