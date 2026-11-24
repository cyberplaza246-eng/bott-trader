p='code/fb_worker.py'
s=open(p,encoding='utf-8').read().replace('\r\n','\n')
old="""            rec['sweep']=sd
            if sd in ('BUY','SELL'):
                r=et.get_trading_signal(dfe,'MNQ')
                m=r['models']"""
new="""            rec['sweep']=sd
            r=et.get_trading_signal(dfe,'MNQ')
            if r['signal'] in ('BUY','SELL'):
                m=r['models']"""
assert old in s
s=s.replace(old,new)
open(p,'w',encoding='utf-8').write(s)
print('ok')
