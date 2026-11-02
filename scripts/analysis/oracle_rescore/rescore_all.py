import pandas as pd, numpy as np, pyreadr
from scipy.stats import kendalltau
S='/mnt/user-data/uploads/c/SHAP_imbalance/results/'
O='/mnt/user-data/uploads/c/SHAP_imbalance/results/oracle_stability/'
sig=list(pyreadr.read_r(S+'full_sim_v5_signed.rds').values())[0]
met=list(pyreadr.read_r(S+'full_sim_v5_metrics.rds').values())[0]
oi=pd.read_csv(O+'oracle_importance_full.csv')
osum=pd.read_csv(O+'oracle_summary_full.csv')
print('oracle rows', len(oi), 'cell_keys', oi.cell_key.nunique(), 'seeds', sorted(oi.seed.unique()))
print('metrics cells', met.cell_key.nunique(), 'signed cells', sig.cell_key.nunique())
feats=['x%d'%i for i in range(1,21)]
sig_feats=['x%d'%i for i in range(1,8)]
# finite share vectors
f=sig[['cell_key','rep','strategy','feature','share_fin','total_fin']].copy()
fin=f.pivot_table(index=['cell_key','rep','strategy'], columns='feature', values='share_fin')[feats]
tot_fin=f.groupby(['cell_key','rep','strategy'])['total_fin'].first()
# oracle vectors per (cell_key, seed)
oracle=oi.pivot_table(index=['cell_key','seed'], columns='feature', values='share')[feats]
oracle_imp=oi.pivot_table(index=['cell_key','seed'], columns='feature', values='imp')[feats]
oracle_tot=osum.set_index(['cell_key','seed'])['total_imp']
# check oracle total equals sum of imp
chk=(oracle_imp.sum(axis=1)-oracle_tot.reindex(oracle_imp.index)).abs().max()
print('max |sum(imp)-total_imp| oracle:', chk)
rows=[]
for (ck,seed),ov in oracle.iterrows():
    o=ov.values; ot=oracle_tot.loc[(ck,seed)]
    sub=fin.xs(ck, level='cell_key')
    for (rep,strat),v in sub.iterrows():
        pass
    idx=sub.index
    V=sub.values
    for k,(rep,strat) in enumerate(idx):
        v=V[k]
        t20=kendalltau(v,o)[0]
        si=[feats.index(x) for x in sig_feats]
        t7=kendalltau(v[si],o[si])[0]
        sL1=ot*np.abs(v-o).sum()
        rows.append((ck,seed,rep,strat,t20,t7,sL1))
R=pd.DataFrame(rows,columns=['cell_key','seed','rep','strategy','tau20','tau7','sL1'])
R.to_pickle('/tmp/claude-0/-home-claude/42f550b3-6df1-57e9-802c-958d5876f7f7/scratchpad/oracle/rescore_long.pkl')
# verification: seed 7777 vs stored metrics kendall
m=met[['cell_key','rep','strategy','kendall','l1']].copy()
j=R[R.seed==7777].merge(m,on=['cell_key','rep','strategy'])
print('verify tau20 vs stored kendall: n',len(j),' max abs diff', (j.tau20-j.kendall).abs().max(), ' median', (j.tau20-j.kendall).abs().median())
print(j[['tau20','kendall']].head())
