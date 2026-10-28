import pandas as pd,numpy as np
df=pd.read_pickle('tr.pkl');df=df[df.t>='2024-10-01'].copy()
df['tr']=df.t<'2025-10-01'
def st(x):
    if len(x)==0:return '-'
    eq=x.pnl.cumsum();mdd=(eq-eq.cummax()).min()
    return f"n={len(x)} win={100*(x.pnl>0).mean():.0f}% $/t={x.pnl.mean():.1f} tot={x.pnl.sum():.0f} mdd={mdd:.0f}"
for p,g in [('TRAIN',df[df.tr]),('TEST',df[~df.tr]),('ALL',df)]: print(p,st(g))
feats=['dret','cloc','gap','r5','rv','dhi20','atr15','adx','s15','prev']
for f in feats:
    q=df[df.tr][f].quantile([.33,.67]).values
    lab=lambda v: 'lo' if v<=q[0] else ('mid' if v<=q[1] else 'hi')
    df['b']=df[f].map(lambda v: 'na' if pd.isna(v) else lab(v))
    print(f'\n{f} cuts {np.round(q,4)}')
    for b in ['lo','mid','hi','na']:
        print(f'  {b:3}| TR {st(df[df.tr&(df.b==b)])} || TE {st(df[~df.tr&(df.b==b)])}')
for f in ['dow','sunday','h1up','monthend']:
    print('\n',f)
    for v in sorted(df[f].unique()): print(f'  {v}| TR {st(df[df.tr&(df[f]==v)])} || TE {st(df[~df.tr&(df[f]==v)])}')
