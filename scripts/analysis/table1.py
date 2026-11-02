import pandas as pd, numpy as np, pyreadr
from scipy.stats import spearmanr
R='../../results/'
m=list(pyreadr.read_r(R+'full_sim_v4_metrics.rds').values())[0]
m=m[m.error.isna()].copy()
f=pd.read_pickle('fit_level_final.pkl')
S=['none','class_weight','smote','adasyn','random_over','random_under']
cols=['kendall','spearman','rbo','jaccard_k5','jaccard_k7','jaccard_k10','recall_k5','recall_k7','l1','l2','cosine']
M=m.groupby('strategy')[cols].mean().reindex(S)
M['gini_corr']=f.groupby('strategy').gini_corr.mean().reindex(S)
M['sL1']=f.groupby('strategy').scale_matched_l1.mean().reindex(S)
LOW=['l1','l2','sL1']
rk=pd.DataFrame({c:M[c].rank(ascending=(c in LOW)).astype(int) for c in M.columns})
print(M.round(3).T.to_string()); print(); print(rk.T.to_string())
M.to_pickle('table1_means.pkl'); rk.to_pickle('table1_ranks.pkl')
# within-cell strategy-rank correlations between metric families
rows=[]
for ck in m.cell_key.unique():
    mm=m[m.cell_key==ck].groupby('strategy')[['kendall','l1','l2','cosine','spearman']].mean().reindex(S)
    mm['sL1']=f[f.cell_key==ck].groupby('strategy').scale_matched_l1.mean().reindex(S)
    rows.append(mm)
def wc(a,b):
    v=[spearmanr(r[a].rank(ascending=(a not in LOW)),r[b].rank(ascending=(b not in LOW))).correlation for r in rows]
    return np.nanmean(v), np.nanstd(v)/np.sqrt(len(v))
print('\nWithin-cell correlation of strategy ranks (mean over 24 cells, SE):')
for a,b in [('kendall','spearman'),('kendall','l1'),('kendall','sL1'),('l1','sL1'),('kendall','cosine'),('kendall','l2'),('l1','cosine')]:
    mu,se=wc(a,b); print(f'{a:9s} vs {b:9s}: {mu:+.2f} (SE {se:.2f})')
