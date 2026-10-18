"""Scientific figures built only from the final independent aggregate scores."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/carbon-path-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'figures'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
    'axes.spines.right':False,'svg.hashsalt':'carbon-path-20261019','savefig.facecolor':'white'})
names={'rw':'Random walk','summary_state':'Joint summary state (candidate)','state_marginal':'Separate horizon scores',
       'state_price':'No future range supervision','state_step1':'One-step training gradients',
       'gru_full':'GRU, all historical fields','free_path':'Free daily joint decoder',
       'drift':'Random walk with drift','gru_price':'GRU, price history',
       'hgb_full':'Histogram boosting, all fields','ridge_price':'Ridge, price history','ridge_full':'Ridge, all fields'}
metrics=pd.read_csv(ROOT/'results/mse_by_cell.csv')
m=metrics[metrics.estimate=='mean_ensemble'].copy()
scores=m.groupby('family').mse_ratio_to_rw.mean().sort_values()
comparisons=pd.read_csv(ROOT/'results/comparisons.csv')
c=comparisons[(comparisons.scope=='overall')&(comparisons.block_length==20)].set_index('comparator')

fig,(a,b)=plt.subplots(1,2,figsize=(13.0,5.8),gridspec_kw={'width_ratios':[1.15,1]})
colors=['#111827' if f=='rw' else '#1763A6' if f=='summary_state' else '#8995A6' for f in scores.index]
y=np.arange(len(scores))
a.scatter(scores.values,y,c=colors,s=45,zorder=3)
a.axvline(1,color='#111827',linestyle='--',linewidth=1)
a.set_yticks(y,[names[f] for f in scores.index]);a.invert_yaxis()
a.set_xlim(.993,1.128);a.set_xticks([1.,1.04,1.08,1.12])
a.grid(axis='x',alpha=.2);a.set_xlabel('Mean cell MSE / random-walk MSE')
a.set_title('A. Price prediction: random walk remains best',loc='left',fontweight='bold',fontsize=11)
for yy,(f,v) in enumerate(scores.items()):a.text(v+.002,yy,f'{v:.4f}',va='center',fontsize=8)
controls=['rw','gru_full','free_path','state_price','state_marginal','state_step1']
for j,f in enumerate(controls):
    r=c.loc[f];col='#B53C42' if f=='rw' else '#1763A6'
    b.errorbar(r.ratio,j,xerr=[[r.ratio-r.ci_lower],[r.ci_upper-r.ratio]],fmt='o',color=col,
               elinewidth=1.8,capsize=3,markersize=5)
b.axvline(1,color='#111827',linestyle='--',linewidth=1)
b.set_yticks(np.arange(len(controls)),[names[f] for f in controls]);b.invert_yaxis()
b.set_xlim(.965,1.102);b.set_xlabel('Candidate / comparator MSE ratio with 95% CI')
b.grid(axis='x',alpha=.2);b.set_title('B. Paired comparisons: primary block length 20',loc='left',fontweight='bold',fontsize=11)
fig.suptitle('Reported EUA price-range learning: predefined development experiment',fontweight='bold',fontsize=14,y=1.01)
fig.text(.5,-.04,'Two author sources of the same EUA market; 6 chronological folds; 1,151 origins; h = 1, 5, 20 observed records.\n'
         'Point estimates average 18 source-fold-horizon MSE ratios. CIs: 2,000 paired moving-block replicates. Lower is better.',
         ha='center',fontsize=9,color='#475569')
fig.tight_layout(w_pad=2.7)
for ext in ('svg','png'):fig.savefig(OUT/f'performance_overview.{ext}',dpi=200,bbox_inches='tight',metadata={'Date':None} if ext=='svg' else {})
plt.close(fig)

candidate=m[m.family=='summary_state'].copy();candidate['key']=candidate.source+'/'+candidate.fold
order=['quantum/Q1','quantum/Q2','quantum/Q3','ding/D1','ding/D2','ding/D3']
matrix=candidate.pivot(index='key',columns='horizon',values='mse_ratio_to_rw').loc[order,[1,5,20]]
fig,ax=plt.subplots(figsize=(7.7,4.9))
im=ax.imshow(matrix.to_numpy(),cmap='RdBu_r',norm=TwoSlopeNorm(vmin=.75,vcenter=1,vmax=1.35),aspect='auto')
ax.set_xticks([0,1,2],['h = 1','h = 5','h = 20'])
ax.set_yticks(range(6),['Quantum Q1 (initialization selected)','Quantum Q2','Quantum Q3 (initialization selected)',
    'Ding: 2022 H2 (initialization selected)','Ding: 2023 H1','Ding: 2023 H2, matured origins'])
for i in range(6):
    for j in range(3):
        value=matrix.iloc[i,j];ax.text(j,i,f'{value:.4f}',ha='center',va='center',
               color='white' if abs(value-1)>.19 else '#17212E',fontsize=11,fontweight='bold')
ax.set_title('Candidate price MSE by chronological fold and horizon',loc='left',fontweight='bold',pad=15)
ax.set_xlabel('Horizon counts observed source records; Quantum dates are unavailable')
cb=fig.colorbar(im,ax=ax,fraction=.045,pad=.045);cb.set_label('MSE / random-walk MSE')
fig.text(.5,-.02,'Only Ding 2023 H1 improves at the fold level. Three folds exactly retain random-walk mean forecasts.\n'
          'Fold membership and all model selection decisions were fixed without test-score tuning.',ha='center',fontsize=9,color='#475569')
fig.tight_layout()
for ext in ('svg','png'):fig.savefig(OUT/f'fold_horizon_results.{ext}',dpi=200,bbox_inches='tight',metadata={'Date':None} if ext=='svg' else {})
plt.close(fig)
print('Created four scientific figure files from final aggregate tables.')
