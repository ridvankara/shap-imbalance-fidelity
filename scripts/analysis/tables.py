import pandas as pd, numpy as np
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
rng=np.random.default_rng(7)
f=pd.read_pickle('fit_level.pkl'); ss=pd.read_pickle('sign_level.pkl')
f=f.dropna(subset=['kendall'])
S=['none','class_weight','smote','adasyn','random_over','random_under']
def boot_mean(df,col,B=1000):
    reps=df.rep.unique(); vals=df.groupby('rep')[col].mean()
    m=vals.mean(); bs=[vals.loc[rng.choice(reps,len(reps))].mean() for _ in range(B)]
    return m,np.percentile(bs,2.5),np.percentile(bs,97.5)
print('== 1. Overall means by strategy (fit-level; CI by bootstrap over replications)')
rows=[]
for s in S:
    d=f[f.strategy==s]; r={'strategy':s,'n_fits':len(d)}
    for c in ['kendall','spearman','rbo','jaccard_k5','recall_k5','l1','l2','cosine','scale_matched_l1','share_l1','gini_corr','scale_ratio']:
        mu,lo,hi=boot_mean(d,c); r[c]=f"{mu:.3f} [{lo:.3f},{hi:.3f}]" if c in['kendall','scale_matched_l1','l1'] else f"{mu:.3f}"
    rows.append(r)
print(pd.DataFrame(rows).set_index('strategy').T.to_string())
print('\n== 2. Kendall by n x strategy (mean)')
t=f.pivot_table(index='strategy',columns='n',values='kendall',aggfunc='mean').reindex(S); print(t.round(3))
print('spread (max-min across strategies) per n:', (t.max()-t.min()).round(3).to_dict())
print('n effect (n=10000 minus n=500, averaged over strategies):', round((t[10000]-t[500]).mean(),3))
print('\n== 3. Kendall by prevalence x strategy (n=500)')
for nn in [500,2000,10000]:
    t2=f[f.n==nn].pivot_table(index='strategy',columns='prev',values='kendall',aggfunc='mean').reindex(S)
    print('n=',nn); print(t2.round(3)); print('spread', (t2.max()-t2.min()).round(3).to_dict())
print('\n== 4. Correlation structure at n=2000: Kendall by corr x strategy')
t3=f[f.n==2000].pivot_table(index='strategy',columns='corr',values='kendall',aggfunc='mean').reindex(S)
t3['diff_block_minus_indep']=t3['block']-t3['independent']; print(t3.round(3))
print('\n== 5. Raw L1 / scale-matched L1 / scale ratio / null share by strategy x n')
for c in ['l1','scale_matched_l1','scale_ratio','null_share_f']:
    print(c); print(f.pivot_table(index='strategy',columns='n',values=c,aggfunc='mean').reindex(S).round(3))
print('\n== 6. Sign metrics (signal features, from per-feature rows)')
sg=ss.groupby(['strategy','sig']).agg(flip=('flip','mean'),agree=('agree','mean'),wagree=('wag','mean'),top50=('t50','mean')).round(3)
print(sg.reindex(S,level=0))
print('\n== 7. Sign flips of mean-phi: share of features flipped per fit by strategy (all features)')
print(ss.groupby('strategy').flip.mean().reindex(S).round(4))
f.to_pickle('fit_level_clean.pkl')
