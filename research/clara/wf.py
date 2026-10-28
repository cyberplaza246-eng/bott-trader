import pandas as pd,numpy as np,itertools,pickle
from filt import FILTERS,CUTS,EXITS,go
lo,hi=pd.Timestamp('2024-09-25'),pd.Timestamp('2026-09-20')
allt={}
for k in itertools.product(FILTERS,CUTS,EXITS):
    t=go(FILTERS[k[0]],k[1],k[2],lo,hi); allt[k]=t
pickle.dump(allt,open('wf_all.pkl','wb'))
