import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
f=pd.read_pickle('fit_level_final.pkl'); ss=pd.read_pickle('sign_level.pkl')
import pyreadr
m=list(pyreadr.read_r('../../results/full_sim_v4_metrics.rds').values())[0]
m=m[m.error.isna()&m.kendall.notna()].copy()
m['n']=m.cell_key.str.extract(r'_n(\d+)$')[0].astype(int)
S=['none','class_weight','smote','adasyn','random_over','random_under']
LAB={'none':'None','class_weight':'Class weight','smote':'SMOTE','adasyn':'ADASYN','random_over':'Random over','random_under':'Random under'}
COL={'none':'#1b1b1b','class_weight':'#7f7f7f','smote':'#1f77b4','adasyn':'#6baed6','random_over':'#d62728','random_under':'#ff9896'}
MK={'none':'o','class_weight':'s','smote':'^','adasyn':'v','random_over':'D','random_under':'x'}
plt.rcParams.update({'font.size':9,'font.family':'DejaVu Sans'})
D='figures/'
# Fig 1: Kendall vs n
fig,ax=plt.subplots(figsize=(5.2,3.6))
for s in S:
    g=m[m.strategy==s].groupby('n').kendall
    mu=g.mean(); se=g.std()/np.sqrt(g.count())*1.96
    ax.errorbar(mu.index,mu.values,yerr=se.values,label=LAB[s],color=COL[s],marker=MK[s],capsize=2,lw=1.2,ms=4)
ax.set_xscale('log'); ax.set_xticks([500,2000,10000]); ax.set_xticklabels(['500','2,000','10,000'])
ax.set_xlabel('Training sample size n (log scale)'); ax.set_ylabel('Kendall τ vs oracle (mean, 95% CI)')
ax.legend(frameon=False,fontsize=7.5,ncol=2); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(D+'fig1_kendall_by_n.png',dpi=300); plt.close(fig)
# Fig 2: Pareto panels (pooled means)
M=f.groupby('strategy')[['kendall','l1','scale_matched_l1']].mean().reindex(S)
fig,axs=plt.subplots(1,2,figsize=(7.2,3.3))
for ax,xc,xl,title in [(axs[0],'l1','Raw L1 distance (lower is better)','(a) Raw magnitude'),(axs[1],'scale_matched_l1','Scale-matched L1 (lower is better)','(b) Scale-matched magnitude')]:
    for s in S:
        ax.scatter(M.loc[s,xc],M.loc[s,'kendall'],color=COL[s],marker=MK[s],s=38,label=LAB[s])
        OFF={'none':(6,6),'class_weight':(-10,-14),'smote':(-22,6),'adasyn':(4,-10),'random_over':(6,4),'random_under':(4,4)}
        ax.annotate(LAB[s],(M.loc[s,xc],M.loc[s,'kendall']),textcoords='offset points',xytext=OFF[s],fontsize=7)
    ax.set_xlabel(xl); ax.set_ylabel('Kendall τ (higher is better)'); ax.set_title(title,fontsize=9); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(D+'fig2_pareto_pooled.png',dpi=300); plt.close(fig)
# Fig 3: sign agreement, signal features (observation level) and |phi|-weighted
sg=ss[ss.sig==True].groupby('strategy')[['agree','wag']].mean().reindex(S)
nl=ss[ss.sig==False].groupby('strategy')[['agree']].mean().reindex(S)
fig,ax=plt.subplots(figsize=(5.4,3.4))
x=np.arange(len(S)); w=.36
ax.bar(x-w/2,sg.agree,w,color='#4c72b0',label='Signal features: observation-level sign agreement')
ax.bar(x+w/2,sg.wag,w,color='#dd8452',label='Signal features: |φ|-weighted sign agreement')
ax.axhline(nl.agree.mean(),color='k',ls='--',lw=.8); ax.text(5.45,nl.agree.mean()+.01,'Null features (chance ≈ 0.50)',fontsize=7,ha='right')
ax.set_xticks(x); ax.set_xticklabels([LAB[s] for s in S],rotation=25,ha='right',fontsize=7.5)
ax.set_ylim(0.4,1.0); ax.set_ylabel('Sign agreement with oracle'); ax.legend(frameon=False,fontsize=6.8,loc='lower left'); ax.grid(axis='y',alpha=.25)
fig.tight_layout(); fig.savefig(D+'fig3_sign_agreement.png',dpi=300); plt.close(fig)
print('ok')
