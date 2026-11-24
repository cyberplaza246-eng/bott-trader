import json
for fn in ['out/results_og.jsonl','out/results_og_stress.jsonl']:
  for l in open(fn):
    r=json.loads(l); c=r['res'].get('c4',{})
    a,t,y=c.get('all',{}),c.get('train',{}),c.get('y2026',{})
    print(f"{r['name']:24s} n{a.get('n')} net{a.get('net',0):9.0f} pf{a.get('pf',0):.2f} dd{a.get('maxdd',0):7.0f} | train {t.get('net',0):8.0f} pf{t.get('pf',0):.2f} | 2026 {y.get('net',0):8.0f} pf{y.get('pf',0):.2f} | green {c.get('months_green')}/{c.get('months')}")
