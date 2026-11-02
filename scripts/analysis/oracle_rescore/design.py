import pandas as pd, numpy as np
R=pd.read_pickle('rescore_long.pkl')
strat=['none','class_weight','smote','adasyn','random_over','random_under']
R=R[R.strategy.isin(strat)]
rng=np.random.default_rng(1)
print("== Design-level means by oracle seed ==")
rows=[]
for s,g in R.groupby('seed'):
    m=g.groupby('strategy')[['tau20','tau7','sL1']].mean().reindex(strat)
    m['seed']=s; rows.append(m.reset_index())
M=pd.concat(rows)
pv=M.pivot(index='strategy',columns='seed',values='tau7').reindex(strat)
pv20=M.pivot(index='strategy',columns='seed',values='tau20').reindex(strat)
pvl=M.pivot(index='strategy',columns='seed',values='sL1').reindex(strat)
print('tau20 (all 20):'); print(pv20.round(3).to_string())
print('tau7 (signal only):'); print(pv.round(3).to_string())
print('sL1:'); print(pvl.round(3).to_string())
print()
print("== Strategy ordering (tau7, high->low) per seed ==")
for s in pv.columns:
    print(s, ' > '.join(pv[s].sort_values(ascending=False).index))
print("== Strategy ordering (tau20, high->low) per seed ==")
for s in pv20.columns:
    print(s, ' > '.join(pv20[s].sort_values(ascending=False).index))
print("== Strategy ordering (sL1, low->high) per seed ==")
for s in pvl.columns:
    print(s, ' < '.join(pvl[s].sort_values().index))
# paired diffs vs none, cluster bootstrap over (cell_key,rep)
def paired(seedsub, metric):
    W=seedsub.pivot_table(index=['cell_key','rep'],columns='strategy',values=metric)
    W=W.dropna()
    out={}
    for st in strat[1:]:
        d=(W[st]-W['none']).values
        bs=[d[rng.integers(0,len(d),len(d))].mean() for _ in range(1000)]
        out[st]=(d.mean(),np.percentile(bs,2.5),np.percentile(bs,97.5))
    return out
print()
print("== Paired differences vs none (per seed, tau7 / tau20 / sL1) ==")
for metric in ['tau7','tau20','sL1']:
    print('--',metric)
    for s,g in R.groupby('seed'):
        p=paired(g,metric)
        print(s, '  '.join(f"{k}:{v[0]:+.4f}[{v[1]:+.4f},{v[2]:+.4f}]" for k,v in p.items()))
