import json
S={}
S['baseline']={'_save':True}
for c in ['trend_day','trend_ovn','bo_day','bo_ovn_0009','bo_ovn_other','mr_day','mr_ovn','onl','e1']: S['drop_'+c]={'drop':[c]}
S['no_revamp_exits']={'revamp':False}
for k in ['off','tight','tight2','wide']: S['lock_'+k]={'lock':k}
S['maxopen1']={'max_open':1}; S['maxopen2']={'max_open':2}
S['tp_trend1.0']={'trend_r':1.0}; S['tp_trend2.0']={'trend_r':2.0}; S['tp_bo1.5']={'bo_r':1.5}; S['tp_bo2.5']={'bo_r':2.5}; S['tp_bo3.0']={'bo_r':3.0}
S['day_only_main']={'drop':['trend_ovn','bo_ovn_0009','bo_ovn_other','mr_ovn']}
S['day_only_all']={'drop':['trend_ovn','bo_ovn_0009','bo_ovn_other','mr_ovn','e1']}
S['slip2_baseline']={'slip':0.5}
S['comm224_check']={'comm':2.24}
json.dump(S,open('specs_p1.json','w'),indent=0)
print(len(S))
