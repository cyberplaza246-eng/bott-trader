from sr_globex import *
import sr_pub as P
cc=P.c; oo=P.o
for sn in SETS:
    ev=SETS[sn]; mv=np.array([e['mv'] for e in ev]); a=np.array([e['i'] for e in ev])
    r1=np.array([cc[i]-oo[i] for i in a]); r5=np.array([cc[i+4]-oo[i] for i in a]); r30=np.array([cc[i+29]-oo[i] for i in a])
    print(sn,'events',len(ev),'mean|gap|',round(np.abs(mv).mean(),1),'corr(gap,1m move)',round(float(np.corrcoef(mv,r1)[0,1]),2),'corr(gap,5m)',round(float(np.corrcoef(mv,r5)[0,1]),2),'corr(gap,30m)',round(float(np.corrcoef(mv,r30)[0,1]),2),'same sign 1m',round(float((np.sign(mv)==np.sign(r1)).mean()),2),'mean 30m move in gap dir',round(float((np.sign(mv)*r30).mean()),1))