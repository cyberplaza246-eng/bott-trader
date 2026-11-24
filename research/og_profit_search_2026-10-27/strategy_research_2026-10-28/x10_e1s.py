import os,sys,pickle
sys.path.insert(0,os.path.abspath('..')); sys.path.insert(0,os.path.abspath('../code'))
import sr_x10_lib as L, pandas as pd, numpy as np
E=L.E; B=L.B
d=E.load(); m5=d['m5']
b=B.featured(m5); c=B.candidate_rows(b)
for k,v in c.items(): print(k,len(v))
e1s=c['e1_short_6pm']; print(e1s.head(3).to_string()); print(d['adds']['addon_e1'].head(3).to_string())
print('addon_e1 n',len(d['adds']['addon_e1']))
e1s.to_pickle('x10_e1_short.pkl')
print(B.ax.E1_FIRST_MIN,B.ax.E1_LAST_MIN,B.ax.E1_RSI_MIN,B.ax.E1_MOVE_ATR_MIN,B.ax.STOP_ATR,B.ax.TARGET_R,B.ax.MAX_BARS)
# check my rebuild of e1 long equals d['adds']
a=B.addon_rows(b)['addon_e1']; print('rebuilt long n',len(a),'match idx',a.index.equals(d['adds']['addon_e1'].index))
