p='code/backtest_revamp_live_stack.py'
s=open(p,encoding='utf-8').read()
a="hit_tp = (h1[j] >= p.tp) if p.dir > 0 else (l1[j] <= p.tp)"
assert a in s
s=s.replace(a,"hit_tp = (h1[j] >= p.tp + TP_THROUGH) if p.dir > 0 else (l1[j] <= p.tp - TP_THROUGH)")
s=s.replace("COMM_RT = 3.0\n","COMM_RT = 3.0\nTP_THROUGH = 0.0   # (added) price must trade this far THROUGH the TP limit to fill\n")
open(p,'w',encoding='utf-8').write(s)
e=open('rv_engine.py',encoding='utf-8-sig').read()
e=e.replace("hours_drop=(),only_dir=None)","hours_drop=(),only_dir=None,tp_through=0.0,ovn_slip=0)")
e=e.replace("B.COMM_RT=s['comm']; B.SLIP=s['slip']","B.COMM_RT=s['comm']; B.SLIP=s['slip']; B.TP_THROUGH=s['tp_through']")
e=e.replace("B.Rules(overnight_sleeves='all',max_open=s['max_open'])","B.Rules(overnight_sleeves='all',max_open=s['max_open'],ovn_extra_slip_ticks=s['ovn_slip'])")
open('rv_engine.py','w',encoding='utf-8').write(e)
print('ok')
