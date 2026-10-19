"""Publication-oriented static figures from completed public aggregate results."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'figures'
BLUE='#0072B2';ORANGE='#D55E00';GRAY='#667085'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'axes.titleweight':'bold','figure.facecolor':'white',
                     'savefig.facecolor':'white','svg.fonttype':'none'})


def save(fig,name):
    OUT.mkdir(parents=True,exist_ok=True)
    fig.savefig(OUT/(name+'.png'),dpi=200,bbox_inches='tight')
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight')
    plt.close(fig)


def synthetic():
    data=json.loads((ROOT/'results/synthetic/SUMMARY.json').read_text())
    groups=data['groups']
    scenario_order=list(dict.fromkeys(x['scenario'] for x in groups))
    names={'conditional_persistence':'Conditional persistence (exact)',
           'constant_persistence':'Constant persistence (exact)',
           'leverage_misspecified':'Leverage stress (Monte Carlo)'}
    fig,ax=plt.subplots(figsize=(9,6.2))
    rows=sorted(groups,key=lambda x:(scenario_order.index(x['scenario']),x['n']))
    labels=[]
    for i,row in enumerate(rows):
        values=np.asarray(list(row['per_seed_primary_relative'].values()))*100
        col=BLUE if values.mean()<0 else ORANGE
        ax.scatter(values,i+np.linspace(-.12,.12,len(values)),s=15,color=col,alpha=.28,zorder=2)
        ax.plot([values.min(),values.max()],[i,i],color=col,lw=1.3,alpha=.65)
        ax.scatter([values.mean()],[i],s=52,color=col,edgecolor='white',linewidth=.6,zorder=3)
        labels.append(names.get(row['scenario'],row['scenario'].replace('_',' '))+f"\n n = {row['n']:,}")
    ax.set_yticks(range(len(rows)),labels);ax.invert_yaxis()
    ax.axvline(0,color=GRAY,lw=1,ls='--');ax.grid(axis='x',alpha=.18)
    ax.set_xlabel('Relative CRPS to the joint-loss constant model (%)  |  lower is better')
    ax.set_title('Synthetic identification of conditional persistence',loc='left',pad=16)
    fig.text(.02,.01,'Dots: mean and 20 training seeds; horizontal segments: seed range, not confidence intervals.\n'
             'Fixed mixture weights and scale. Primary/negative-control expectations are analytic; leverage stress uses shared Monte Carlo truth.',
             fontsize=8,color=GRAY)
    fig.tight_layout(rect=(0,.07,1,1))
    save(fig,'01_synthetic_identification')


def heatmap(summary):
    key='GRU16__conditional_joint005__vs__GRU16__constant_joint005'
    panels=summary['strata']['development']['panels']+summary['strata']['additional_shenzhen']['panels']
    vals=[]
    for panel in panels:
        stratum='additional_shenzhen' if panel.startswith('SHENZHEN') else 'development'
        contrasts=summary['comparisons'][key][stratum]
        vals.append([100*contrasts['h1_secondary' if h==1 else 'h5_h20']['cell_relative_values'][f'{panel}_h{h}'] for h in (1,5,20)])
    values=np.asarray(vals)
    scale=max(float(np.abs(values).max()),.01)
    fig,ax=plt.subplots(figsize=(7.4,8.4))
    im=ax.imshow(values,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-scale,vcenter=0,vmax=scale),aspect='auto')
    for i in range(len(panels)):
        for j in range(3):
            c='white' if abs(values[i,j])>.62*scale else '#17212B'
            ax.text(j,i,f'{values[i,j]:+.3f}%',ha='center',va='center',color=c,fontsize=9)
    labels=[p.replace('EUA_LONG','EUA').replace('SHENZHEN_PLOS','SHENZHEN').replace('_',' ') for p in panels]
    ax.set_yticks(range(len(panels)),labels)
    ax.set_xticks(range(3),['1 quote-step\nsecondary','5 quote-steps\nprimary','20 quote-steps\nprimary'])
    ax.axhline(10.5,color='black',lw=1.8)
    ax.set_title('Conditional / constant persistence: cell-level CRPS',loc='left',pad=16)
    fig.colorbar(im,ax=ax,shrink=.8,pad=.035,label='Relative CRPS change (%)')
    fig.text(.02,.015,'Three seed scores averaged first; negative values favor conditional persistence.\n'
             'The final two rows are first-model-use Shenzhen years, reported separately from the 11 development panels.',fontsize=8,color=GRAY)
    fig.tight_layout(rect=(0,.055,1,1))
    save(fig,'02_market_cell_effects')


def forest(summary):
    comparisons=summary['comparisons']
    rows=[('GRU: conditional vs constant',summary['primary_GRU_development']),
          ('GRU: conditional vs equal-capacity scale',summary['capacity_GRU_development']),
          ('GRU: path auxiliary vs endpoint-only',comparisons['GRU16__conditional_joint005__vs__GRU16__conditional_endpoint']['development']['h5_h20']),
          ('GRU: shared vs independent horizon heads',comparisons['GRU16__conditional_endpoint__vs__GRU16__independent_horizon_endpoint']['development']['h5_h20']),
          ('MLP: conditional vs constant',summary['replication_MLP_development']),
          ('Shenzhen GRU: conditional vs constant',summary['additional_GRU_shenzhen'])]
    fig,ax=plt.subplots(figsize=(10.3,5.6))
    for i,(label,row) in enumerate(rows):
        mean=100*row['mean_relative_CRPS_minus_one'];low,high=100*np.asarray(row['bootstrap_nominal95_relative_CRPS'])
        col=BLUE if mean<0 else ORANGE
        ax.plot([low,high],[i,i],color=col,lw=2)
        ax.scatter(mean,i,s=62,color=col,zorder=3,edgecolor='white',linewidth=.8)
        ax.annotate(f"{mean:+.3f}%   ({row['better_cells']}/{row['cells']} better)",(high,i),xytext=(8,0),textcoords='offset points',va='center',fontsize=9)
    ax.set_yticks(range(len(rows)),[a for a,b in rows]);ax.invert_yaxis()
    ax.axvline(0,color=GRAY,ls='--',lw=1);ax.grid(axis='x',alpha=.18)
    ax.axhline(4.5,color=GRAY,lw=.6,alpha=.5)
    low,high=ax.get_xlim();ax.set_xlim(low,high+.36*(high-low))
    ax.set_xlabel('Equal-cell relative CRPS change (%)  |  lower is better')
    ax.set_title('Matched comparisons and mechanism controls',loc='left',pad=16)
    fig.text(.02,.012,'Intervals: 1,999 paired circular block resamples, block length 40, nominal 95%.\n'
             'First five rows: 22 development cells. Final row: 4 Shenzhen cells. Independent horizon heads have 17 extra parameters.\n'
             'These intervals condition on fixed panels; they are not multiplicity-adjusted or an external-validation claim.',fontsize=8,color=GRAY)
    fig.tight_layout(rect=(0,.105,1,1))
    save(fig,'03_matched_comparisons')


def context():
    frame=pd.read_csv(ROOT/'results/cross_benchmark/FAMILY_SUMMARY.csv')
    names={'CP_conditional_joint005':'Conditional persistence + path',
           'CP_constant_joint005':'Constant persistence + path',
           'CP_scale_capacity_joint005':'Equal-capacity scale + path',
           'CP_conditional_endpoint':'Conditional persistence, endpoint',
           'CP_independent_horizon_endpoint':'Independent horizon heads',
           'gas_gru_gmm':'GAS-Norm + GRU-GMM','ewma_gru_gmm':'EWMA + GRU-GMM','global_gru_gmm':'Global + GRU-GMM',
           'gas_no_neural':'GAS-Norm, fixed GMM','ewma_no_neural':'EWMA, fixed GMM','global_no_neural':'Global, fixed GMM',
           'gas_empirical101':'GAS-Norm, 101 empirical atoms','ewma_empirical101':'EWMA, 101 empirical atoms',
           'global_empirical101':'Global, 101 empirical atoms','daily_ewma_normal':'Daily EWMA Normal'}
    fig,axes=plt.subplots(1,2,figsize=(11.5,7.8),sharey=True,gridspec_kw={'width_ratios':[1,1]})
    y=np.arange(len(frame)); labels=[names[x] for x in frame.family]
    arrays=[100*frame.normalized_CRPS_relative_daily_normal.values,100*(frame.mean_price_MSE_relative_RW.values-1)]
    for ax,values,title in zip(axes,arrays,['CRPS relative to daily EWMA Normal (%)','Price-mean MSE relative to random walk (%)']):
        ax.barh(y,values,color=[BLUE if v<0 else ORANGE if v>0 else GRAY for v in values],height=.65,alpha=.85)
        ax.axvline(0,color=GRAY,lw=.8);ax.grid(axis='x',alpha=.18)
        for i,v in enumerate(values):
            ax.annotate(f'{v:+.2f}',(v,i),xytext=(4 if v>=0 else -4,0),textcoords='offset points',
                        ha='left' if v>=0 else 'right',va='center',fontsize=8)
        left,right=ax.get_xlim();pad=(right-left)*.22
        ax.set_xlim(left-pad,right+pad);ax.set_xlabel(title,fontsize=9)
        ax.axhline(4.5,color=GRAY,lw=.6,alpha=.6)
    axes[0].set_yticks(y,labels);axes[0].invert_yaxis()
    fig.suptitle('Contextual benchmark on identical development origins',x=.03,ha='left',fontweight='bold',fontsize=13)
    fig.text(.02,.014,'22 fixed market-year/horizon cells; seed scores averaged before equal-cell ratios. Negative values favor the named method.\n'
             'The two panels use different reference forecasts. Representations, horizon coupling, location freedom and total parameter counts differ across families.\n'
             'This is a descriptive system comparison, not an isolated architecture effect or a reproduction of original paper leaderboards.',fontsize=8,color=GRAY)
    fig.tight_layout(rect=(0,.075,1,.96))
    save(fig,'04_contextual_benchmarks')


def main():
    summary=json.loads((ROOT/'results/market/persistence/SUMMARY.json').read_text())
    assert summary['fit_accounting']['neural_fits']==312
    synthetic();heatmap(summary);forest(summary);context()
    inputs=[ROOT/'results/synthetic/SUMMARY.json',ROOT/'results/market/persistence/SUMMARY.json',ROOT/'results/cross_benchmark/FAMILY_SUMMARY.csv']
    metadata={'generated_from_completed_aggregate_results':True,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'inputs':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
              'raw_or_per_origin_price_values_plotted':False,
              'figures':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir()) if p.suffix in ('.png','.svg')}}
    (OUT/'FIGURE_PROVENANCE.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({'figures':4,'formats':['PNG','SVG']}))


if __name__=='__main__':main()
