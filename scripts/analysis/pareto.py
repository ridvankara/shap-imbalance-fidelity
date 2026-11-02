import pandas as pd, numpy as np
rng=np.random.default_rng(11)
f=pd.read_pickle('fit_level_clean.pkl'); ss=pd.read_pickle('sign_level.pkl')
S=['none','class_weight','smote','adasyn','random_over','random_under']
sig=ss[ss.sig==True].groupby(['cell_key','rep','strategy']).agg(sign_sig=('agree','mean'),wsign_sig=('wag','mean')).reset_index()
f=f.merge(sig,on=['cell_key','rep','strategy'],how='left')
def frontier(pts):
    # pts: dict name->(maximise vec); returns non-dominated names. Each value tuple already oriented as 'higher is better'
    names=list(pts); keep=[]
    for a in names:
        dom=False
        for b in names:
            if a==b: continue
            if all(pts[b][i]>=pts[a][i] for i in range(len(pts[a]))) and any(pts[b][i]>pts[a][i] for i in range(len(pts[a]))):
                dom=True;break
        if not dom: keep.append(a)
    return keep
def means(df): return df.groupby('strategy')[['kendall','scale_matched_l1','l1','wsign_sig','sign_sig','gini_corr']].mean()
M=means(f).reindex(S)
print(M.round(4))
def orient(M,cols):
    o=M[cols].copy()
    for c in cols:
        if c in ['scale_matched_l1','l1']: o[c]=-o[c]
    return {k:tuple(o.loc[k,cols]) for k in o.index}
print('\nPooled frontier (kendall, scale-matched L1):', frontier(orient(M,['kendall','scale_matched_l1'])))
print('Pooled frontier (kendall, raw L1):', frontier(orient(M,['kendall','l1'])))
print('Pooled frontier (kendall, scale-matched L1, signal observation sign agreement):', frontier(orient(M,['kendall','scale_matched_l1','sign_sig'])))
print('Pooled frontier (kendall, scale-matched L1, weighted sign agreement):', frontier(orient(M,['kendall','scale_matched_l1','wsign_sig'])))
# per-cell frontiers
cells=f.groupby('cell_key')
rows=[]
for ck,d in cells:
    Mc=means(d).reindex(S).dropna(how='all')
    for name,cols in [('tau_sL1',['kendall','scale_matched_l1']),('tau_rawL1',['kendall','l1']),('tau_sL1_sign',['kendall','scale_matched_l1','sign_sig'])]:
        Mc2=Mc.dropna(subset=cols)
        fr=frontier(orient(Mc2,cols))
        rows.append({'cell':ck,'frame':name,'frontier':','.join(fr)})
R=pd.DataFrame(rows)
for name in R.frame.unique():
    sub=R[R.frame==name]
    cnt={s:int(sub.frontier.str.contains(s).sum()) for s in S}
    print(f'\nPer-cell frontier membership ({name}; out of {len(sub)} cells):',cnt)
# paired differences vs none, bootstrap over reps (paired on cell-rep)
base=f[f.strategy=='none'].set_index(['cell_key','rep'])
def paired(s,col):
    d=f[f.strategy==s].set_index(['cell_key','rep'])
    j=d[[col]].join(base[[col]],rsuffix='_n',how='inner').dropna()
    x=(j[col]-j[col+'_n']).values
    reps=j.index.get_level_values('rep').values
    ur=np.unique(reps)
    bs=[]
    for _ in range(1000):
        samp=rng.choice(ur,len(ur))
        idx=np.concatenate([np.where(reps==r)[0] for r in samp])
        bs.append(x[idx].mean())
    return x.mean(),np.percentile(bs,2.5),np.percentile(bs,97.5),len(x)
print('\nPaired difference vs None: mean [95% CI] (n pairs)')
for col in ['kendall','scale_matched_l1','l1','sign_sig','wsign_sig','gini_corr']:
    for s in S[1:]:
        mu,lo,hi,n=paired(s,col)
        print(f'{col:17s} {s:13s} {mu:+.4f} [{lo:+.4f},{hi:+.4f}] n={n}')
f.to_pickle('fit_level_final.pkl')
