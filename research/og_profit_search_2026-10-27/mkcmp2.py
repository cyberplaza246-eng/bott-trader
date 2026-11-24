s=open('cmp.py',encoding='utf-8').read()
i=s.index("tmap=")
head=s[:i]
head=head.replace("et.lstm_available=False\n","")
tail='''
import random,numpy as np
random.seed(3); idxs=random.sample(range(30000,120000),25)
def run(i,reduced):
    df=b.iloc[i-99:i+1].copy().reset_index(drop=True); FDT._now=b.datetime.iloc[i].to_pydatetime()
    dfe=ta.calculate_indicators(df)
    sw=et.sweep.get_signal(dfe,'MNQ',df_5m=dfe.tail(250).copy(),spread=None)
    sf=sw.get('signal') in ('BUY','SELL')
    et.lstm_available=et.intelligent_available=et.advanced_strats_available=(not reduced) or sf
    r=et.get_trading_signal(dfe,'MNQ')
    return sf,r['signal'],r['confidence'],r.get('regime'),r.get('rsi')
t0=time.time(); full=[run(i,False) for i in idxs]; t1=time.time(); red=[run(i,True) for i in idxs]; t2=time.time()
bad=[(i,a,b_) for i,a,b_ in zip(idxs,full,red) if (a[1],round(a[2],4))!=(b_[1],round(b_[2],4)) or a[3]!=b_[3]]
print('full s/bar',(t1-t0)/25,'reduced s/bar',(t2-t1)/25,'mismatch',len(bad),'nonsweep',sum(1 for a in full if not a[0]),'sigs',sum(1 for a in full if a[1] in('BUY','SELL')))
print(bad[:5])
'''
open('cmp2.py','w',encoding='utf-8').write(head+tail)
