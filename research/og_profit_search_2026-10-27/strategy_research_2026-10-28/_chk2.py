import sys,io,contextlib,json
src=open('sr_gap2.py',encoding='utf-8-sig').read().split('# ---- run ----')[0]
with contextlib.redirect_stdout(io.StringIO()): exec(src,globals())
roll=['2025-03-18','2025-09-16','2025-12-16','2026-03-16','2026-09-14','2026-09-15','2024-12-16','2025-06-16','2025-06-17','2025-03-17']
for D in roll:
    if D in sess:
        k=days.index(D); print(D,'gap',round(sess[D]['op']-sess[days[k-1]]['cl'],1),'prev close',sess[days[k-1]]['cl'],'open',sess[D]['op'])
rr=dfm(run_ev(EV['ny'],'q',.75,90,1))
rr['d']=rr.entry_time.dt.tz_convert(ET).dt.strftime('%Y-%m-%d')
x=rr[rr.d.isin(roll)]
print('trades on roll-ish days',len(x),round(x.pnl.sum())); print(x[['d','dir','pnl']].to_string())
y=rr[~rr.d.isin(roll)]; s=st(y); print('ex-roll-days: n',len(y),'net',round(y.pnl.sum()),'pf',s['pf'],'dd',s['dd'],'train',s['train'],'y26',s['y26'],'folds',s['folds_pos'],'boot_p',s['boot_p'])
