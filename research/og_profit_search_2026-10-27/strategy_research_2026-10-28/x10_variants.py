"""Addendum 10 variants (all re-simulated through the full engine). name -> (group, spec, description)"""
BO,TR,MR,E1='ens_breakout','ens_trend','ens_mean_reversion','addon_e1'
SL={'breakout':[BO],'trend':[TR],'mean_rev':[MR]}
H=lambda h:(h*60,(h+1)*60)
V={}
def add(n,g,spec,desc): V[n]=(g,spec,desc)
add('base','0',{},'baseline')
# G1: 6PM add-on
add('e1_off','G1',dict(drops=[dict(sess='allnight',tags=[E1])]),'remove the 6PM long add-on (addon_e1)')
add('e1_plus_short','G1',dict(extra=['e1_short_6pm']),'keep 6PM long AND add mirror 6PM short (RSI<=40.5 & down >=1.48 ATR)')
add('e1_swap_short','G1',dict(drops=[dict(sess='allnight',tags=[E1])],extra=['e1_short_6pm']),'remove 6PM long, add mirror 6PM short only')
# G2: sleeve x session on/off
for s in ('evening','night','overnight'):
    for sn,tg in SL.items(): add(f'off_{s}_{sn}','G2',dict(drops=[dict(sess=s,tags=tg)]),f'no {sn} entries in {s}')
    add(f'off_{s}_all','G2',dict(drops=[dict(sess=s)]),f'no entries at all in {s}')
add('off_18_0930_all','G2',dict(drops=[dict(sess='allnight')]),'no trading 18:00-09:30 (day only + evening)')
# G3: the earlier overnight-breakout trim and neighbours
add('trim_bo_2_8','G3',dict(drops=[dict(win=(120,480),tags=[BO])]),'cut breakouts 02:00-08:00 (earlier chosen trim)')
add('trim_bo_0_9','G3',dict(drops=[dict(win=(0,540),tags=[BO])]),'cut breakouts 00:00-09:00 (old bo_ovn_0009)')
add('trim_bo_2_930','G3',dict(drops=[dict(win=(120,570),tags=[BO])]),'cut breakouts 02:00-09:30 (= off_overnight_breakout)')
add('trim_all_2_8','G3',dict(drops=[dict(win=(120,480))]),'cut all sleeves 02:00-08:00')
add('trim_bo_3_6','G3',dict(drops=[dict(win=(180,360),tags=[BO])]),'cut breakouts 03:00-06:00 (post-hoc: the two worst hours)')
# G4: direction
for s in ('evening','night','overnight'):
    for d,dn in ((1,'long'),(-1,'short')): add(f'off_{s}_{dn}','G4',dict(drops=[dict(sess=s,dir=d)]),f'no {dn} entries in {s}')
for d,dn in ((1,'long'),(-1,'short')): add(f'off_18_0930_{dn}','G4',dict(drops=[dict(sess='allnight',dir=d)]),f'no {dn} entries 18:00-09:30 (all sleeves)')
for sn,tg in SL.items():
    for d,dn in ((1,'long'),(-1,'short')): add(f'off_18_0930_{sn}_{dn}','G4',dict(drops=[dict(sess='allnight',tags=tg,dir=d)]),f'no {sn} {dn} 18:00-09:30')
for s in ('night','overnight'):
    for d,dn in ((1,'long'),(-1,'short')): add(f'off_{s}_breakout_{dn}','G4',dict(drops=[dict(sess=s,tags=[BO],dir=d)]),f'no breakout {dn} in {s}')
# G5: single-hour drops (all sleeves, ET bar-end hour)
for h in list(range(18,24))+list(range(0,9)): add(f'off_h{h:02d}','G5',dict(drops=[dict(win=H(h))]),f'no entries {h:02d}:00-{h:02d}:59 ET')
add('off_h09_0930','G5',dict(drops=[dict(win=(540,570))]),'no entries 09:00-09:29 ET')
# G6: trend-alignment filter at night (drop night breakout+trend entries that fight the trend)
for f in ('htf576','htf2880','vwap_sign','sma200'): add(f'align_{f}','G6',dict(drops=[dict(sess='allnight',tags=[BO,TR],against=f)]),f'18:00-09:30 breakout/trend only with {f}')
# combos (chosen AFTER seeing singles -> post-hoc)
add('combo_A','G7',dict(drops=[dict(sess='allnight',tags=[E1]),dict(sess='allnight',tags=[TR],dir=-1)]),'6PM long off + no night/overnight trend shorts (post-hoc)')
add('combo_B','G7',dict(drops=[dict(sess='allnight',tags=[E1]),dict(sess='allnight',tags=[TR],dir=-1),dict(sess='allnight',tags=[BO,TR],against='htf2880')]),'combo_A + night breakout/trend only with the 2880-bar trend (post-hoc)')
add('combo_E1htf','G7',dict(drops=[dict(sess='allnight',tags=[E1]),dict(sess='allnight',tags=[BO,TR],against='htf2880')]),'6PM long off + night breakout/trend only with 2880-bar trend (post-hoc)')
