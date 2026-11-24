from sr_dbl import *
import scipy.stats as ss
rows=[]
for q in (0.5,0.6,0.65,0.7,0.75,0.8,0.85,0.9):
    for H in (30,60,90,120,180,240):
        a=dfm(ny(q,H,lo=A0,hi=A1)); b=dfm(ny(q,H,lo=B0,hi=B1)); rows.append((q,H,len(a),round(a.pnl.sum()),len(b),round(b.pnl.sum())))
G=pd.DataFrame(rows,columns=['q','H','nA','netA','nB','netB'])
print('A net grid (rows q, cols H)'); print(G.pivot(index='q',columns='H',values='netA').to_string()); print('B net grid'); print(G.pivot(index='q',columns='H',values='netB').to_string())
print('rank corr A vs B net (48 cells): spearman %.2f'%ss.spearmanr(G.netA,G.netB)[0],' cells positive in A',(G.netA>0).sum(),' in B',(G.netB>0).sum())
print('B net by q (mean over H):',G.groupby('q').netB.mean().round().to_dict())
print('A net by q (mean over H):',G.groupby('q').netA.mean().round().to_dict())
# A-selected rules with H fixed 90, and simple q>=0.65 family average
# reverse direction: freeze on B, score on A
b=G.sort_values('netB',ascending=False).iloc[0]; print('best on B:',b.to_dict())