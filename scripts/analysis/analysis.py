import pyreadr, pandas as pd, numpy as np, json, sys
from scipy.stats import rankdata
rng=np.random.default_rng(2026)
R='../../results/'
m=list(pyreadr.read_r(R+'full_sim_v5_metrics.rds').values())[0]
s=list(pyreadr.read_r(R+'full_sim_v5_signed.rds').values())[0]
m['prev']=m.cell_key.str.extract(r'_p(\d+)_')[0].astype(int)/100
m['n']=m.cell_key.str.extract(r'_n(\d+)$')[0].astype(int)
m['corr']=m.cell_key.str.split('_').str[0]
S=['none','random_over','random_under','smote','adasyn','class_weight']
out={}
# counts
out['rows']=len(m); out['errors']=int(m.error.notna().sum())
ok=m[m.error.isna()]
out['kendall_na']=int(ok.kendall.isna().sum())
out['succ_by_strategy']=ok.groupby('strategy').size().to_dict()
# signed per-fit
s['ad']=(s.share_fin-s.share_or).abs()
s['sig']=s.is_signal.astype(bool)
g=s.dropna(subset=['share_fin']).groupby(['cell_key','rep','strategy'])
f=g.agg(share_l1=('ad','sum'),tot_f=('total_fin','first'),tot_o=('total_or','first'),
        null_f=('null_fin','first'),null_o=('null_or','first'),
        flips=('sign_flip','sum'),nflip_sig=('sign_flip',lambda x: 0)).reset_index()
f['scale_matched_l1']=f.share_l1*f.tot_o
f['scale_ratio']=f.tot_f/f.tot_o
f['null_share_f']=f.null_f/f.tot_f
# feature-level sign metrics
sig=s.dropna(subset=['share_fin'])
ss=sig.groupby(['cell_key','rep','strategy','sig']).agg(flip=('sign_flip','mean'),agree=('obs_sign_agree','mean'),wag=('wagree','mean'),t50=('top50_agree','mean')).reset_index()
f=f.merge(m[['cell_key','rep','strategy','kendall','spearman','rbo','jaccard_k5','jaccard_k7','jaccard_k10','recall_k5','recall_k7','l1','l2','cosine','prev','n','corr']],on=['cell_key','rep','strategy'],how='inner')
# gini correlation: oracle ranks vs finite importance (symmetric average of both directions)
def gini_corr(x,y):
    # Yitzhaki Gini correlation of x with respect to y: cov(x, rank(y)) / cov(x, rank(x))
    rx=rankdata(x); ry=rankdata(y)
    c1=np.cov(x,ry)[0,1]; c2=np.cov(x,rx)[0,1]
    return c1/c2 if c2!=0 else np.nan
gc=[]
for (ck,rp,st),d in sig.groupby(['cell_key','rep','strategy']):
    d=d.sort_values('feature') if 'feature' in d else d
    imp_f=d.share_fin.values; imp_o=d.share_or.values
    gc.append((ck,rp,st,(gini_corr(imp_f,imp_o)+gini_corr(imp_o,imp_f))/2))
gc=pd.DataFrame(gc,columns=['cell_key','rep','strategy','gini_corr'])
f=f.merge(gc,on=['cell_key','rep','strategy'],how='left')
f.to_pickle('./fit_level.pkl')
ss.to_pickle('./sign_level.pkl')
out['degenerate_random_under']=int(len(ok[ok.strategy=='random_under'])-len(f[f.strategy=='random_under']))
out['n_fit_level']=len(f)
json.dump(out,open('./counts.json','w'),indent=1,default=str)
print(json.dumps(out,indent=1,default=str))
