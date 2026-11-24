import json
rows=[json.loads(l) for l in open('out/results_p1.jsonl')]
print(len(rows),'rows')
def f(m): return f"n={m['n']:5d} net={m['net']:9.0f} PF={m['pf']:.2f} wr={m['wr']:.0f} dd={m['dd']:.0f}"
for r in sorted(rows,key=lambda r:r['name']):
    if 'err' in r: print(r['name'],'ERR',r['err'][-200:]); continue
    x=r['res']; print(f"{r['name']:18s} ALL {f(x['all'])} | TRAIN {x['train']['net']:7.0f} dd{x['train']['dd']:5.0f} PF{x['train']['pf']:.2f} | 2026 {x['y2026']['net']:7.0f} dd{x['y2026']['dd']:5.0f} PF{x['y2026']['pf']:.2f} | per {list(x['per'].values())} mg {x['months_green']}/{x['months']}")
