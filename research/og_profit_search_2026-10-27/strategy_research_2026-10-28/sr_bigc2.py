import sr_bigc as B, pandas as pd, numpy as np, json
P=B.P
cands=[('range',dict(N=4),5,'cont','T60'),('range',dict(N=4),5,'cont','T30'),('news',dict(hm=600,R=1.5,nm='1000'),5,'cont','T60'),('news',dict(hm=600,R=1.5,nm='1000'),5,'cont','T30'),('news',dict(hm=600,R=1.5,nm='1000'),5,'cont','T15'),
 ('range',dict(N=4),5,'fade','T60'),('news',dict(hm=600,R=1.5,nm='1000'),5,'fade','T60')]
out=[]
for f,p,tf,mode,ex in cands:
    idx,sd=B.signals(f,p,tf); nm=B.vname(f,p,tf,mode,ex); rec=dict(name=nm)
    for lab,dl,tk in (('base',0,1),('slip3',0,3),('slip4',0,4),('delay1',1,1),('delay2',2,1),('delay3',3,1),('d3s3',3,3)):
        df=P.dfm(B.run(idx,sd,tf,mode,ex,dl,tk).to_dict('records')); s=P.st(df); rec[lab]=s['net']
        if lab=='base':
            rec.update({k:s[k] for k in('n','pf','dd','worst','train','y26','folds_pos','boot_p','t')}); rec['corr_base']=B.corr_base(df)
            c=P.comb(df); rec['port_net']=c['net']; rec['port_dd']=c['dd']; rec['port_wf']=c['wf']
            et=df.entry_time.dt.tz_convert(B.ET); rec['pct_RTH']=round(float(((et.dt.hour*60+et.dt.minute>=570)&(et.dt.hour*60+et.dt.minute<960)).mean()*100)); rec['long%']=round(float((df.dir>0).mean()*100))
    out.append(rec); print(rec,flush=True)
json.dump(out,open('bigc_stress.json','w'),indent=1,default=str)
print('base portfolio',P.st(P.base)['net'],P.st(P.base)['dd'])