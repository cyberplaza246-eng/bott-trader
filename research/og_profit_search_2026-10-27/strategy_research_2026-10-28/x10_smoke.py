import time,sys
import sr_x10_lib as L
t=time.time(); df=L.sim({}); print('base',len(df),round(df.pnl.sum(),1),round(time.time()-t),'s')
t=time.time(); d2=L.sim(dict(drops=[dict(sess='night',tags=['addon_e1'])])); print('no e1 night',len(d2),round(d2.pnl.sum(),1),round(time.time()-t),'s')
print(df.tag.value_counts().to_dict()); print(df.columns.tolist())
