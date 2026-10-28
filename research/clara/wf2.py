import pandas as pd,itertools,pickle
from filt import FILTERS2,CUTS,EXITS,go
lo,hi=pd.Timestamp('2024-09-25'),pd.Timestamp('2026-09-20')
pickle.dump({k:go(FILTERS2[k[0]],k[1],k[2],lo,hi) for k in itertools.product(FILTERS2,CUTS,EXITS)},open('wf_all2.pkl','wb'))
