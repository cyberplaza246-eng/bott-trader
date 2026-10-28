exec(open('combo2.py').read().split("rows=[]")[0])
from report import metrics
for p in [(750,5),(1000,10),(750,10)]:
    t=combine([rev['trend_ovn_max1']],pause=p)
    for per,(a,b) in [('train',TR),('test26',TE)]:
        r=metrics(t[(t.d>=a)&(t.d<=b)],'wo',f'trend_ovn_max1|pause{p}|{per}');br=r.pop('breach_2k_dates');print(json.dumps(r),'breaches',len(br),br[:3])
t=combine([rev['trend_ovn_max1']],pause=(1000,5))
for per,(a,b) in [('train',TR),('test26',TE)]:
    r=metrics(t[(t.d>=a)&(t.d<=b)],'wo',f'trend_ovn_max1|pause1000_5d|{per}');br=r.pop('breach_2k_dates');print(json.dumps(r),'breaches',br)
