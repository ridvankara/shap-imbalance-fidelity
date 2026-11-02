import pandas as pd, numpy as np, pyreadr
from scipy.stats import kendalltau
O='/mnt/user-data/uploads/c/SHAP_imbalance/results/oracle_stability/'
feats=['x%d'%i for i in range(1,21)]; sf=feats[:7]; nf=feats[7:]
oi=pd.read_csv(O+'oracle_importance_full.csv')
R=pd.read_pickle('rescore_long.pkl')
oracle=oi.pivot_table(index=['cell_key','seed'], columns='feature', values='share')[feats]
# 1) Are oracles identical across n for same (corr,prev)?
oi['corr']=oi.cell_key.str.split('_').str[0]; oi['prev']=oi.cell_key.str.split('_').str[1]
g=oi.groupby(['corr','prev','seed'])['share'].nunique()
print('max distinct share values per (corr,prev,seed) across n:', oi.groupby(['corr','prev','seed','feature']).share.nunique().max())
# 2) Oracle-oracle agreement per (corr,prev): mean pairwise tau over seeds
keys=sorted(set(k for k in oracle.index.get_level_values(0)))
seen={}
for ck in keys:
    cp=ck.rsplit('_n',1)[0]
    if cp in seen: continue
    seen[cp]=ck
seeds=[7777,1001,1002,1003,1004,1005]
out=[]
for cp,ck in seen.items():
    V={s:oracle.loc[(ck,s)].values for s in seeds}
    taus20=[];taus7=[];taun=[]
    for i in range(6):
        for j in range(i+1,6):
            a,b=seeds[i],seeds[j]
            taus20.append(kendalltau(V[a],V[b])[0])
            taus7.append(kendalltau(V[a][:7],V[b][:7])[0])
            taun.append(kendalltau(V[a][7:],V[b][7:])[0])
    out.append((cp,np.mean(taus20),np.min(taus20),np.max(taus20),np.mean(taus7),np.mean(taun)))
T=pd.DataFrame(out,columns=['cell_prev_corr','tau20_mean','tau20_min','tau20_max','tau7_mean','tau_null_mean'])
print(T.round(3).to_string())
print('overall oracle-oracle tau20 mean %.3f, tau7 %.3f, null %.3f'%(T.tau20_mean.mean(),T.tau7_mean.mean(),T.tau_null_mean.mean()))
