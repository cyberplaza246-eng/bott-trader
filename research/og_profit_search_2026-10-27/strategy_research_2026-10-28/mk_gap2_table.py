import json
r=json.load(open('gap2_results.json'))
L=['| variant | n | /mo | net | PF | DD | train | 2026 | folds+ | boot_p | slip3 | slip4 | corr | flag |','|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
for k,s in r.items():
    if 'net' not in s: L.append(f"| {k} | {s['n']} | | | | | | | | | | | | n<8 |"); continue
    L.append(f"| {k} | {s['n']} | {s['tpm']} | {s['net']} | {s['pf']} | {s['dd']} | {s['train']} | {s['y26']} | {s['folds_pos']} | {s['boot_p']} | {s['slip3']} | {s['slip4']} | {s['corr']} | {s['flag']} |")
open('GAP2_TABLE.md','w').write('\n'.join(L)); print(len(L))
