"""Scientific figures from completed reported results, without model fitting."""
from pathlib import Path
import json
from collections import defaultdict
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'figures'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,
                     'svg.fonttype':'none','savefig.dpi':220})


def main():
    OUT.mkdir(exist_ok=True)
    conf=json.loads((ROOT/'CONFIRMATION_SUMMARY.json').read_text())
    inference=json.loads((ROOT/'CONFIRMATION_INFERENCE.json').read_text())
    labels={
        'pooled:augment_h32_s1_l0':'Matched augmentation',
        'pooled:kcm_h32_s1_l1':'BCM-GRU (candidate)',
        'pooled:kcm_unblocked_h32_s1_l1':'No pair blocking',
        'pooled:mcssl_h32_s1_l1':'Adapted MCSSL control',
        'pooled:pointwise_h32_s1_l1':'Pointwise consistency',
        'pooled:gru_h32_s1_l0':'Same-width GRU',
        'local:gru_price_h16_s1_l0':'Local price GRU',
        'pooled:gru_price_h16_s0.1_l0':'Development-selected price GRU',
        'pooled:lstm_price_h32_s1_l0':'Shared price LSTM',
        'deterministic:rw':'Random walk',
        'deterministic:stat':'Statistical reference',
        'pooled:ridge_a1':'Pooled ridge',
        'local:patchtst_h16_s1_l0':'Local PatchTST (21-point history)'}
    fig=plt.figure(figsize=(12.8,7.0),layout='constrained')
    gs=fig.add_gridspec(2,2,width_ratios=[1.7,1.0])
    ax=fig.add_subplot(gs[:,0])
    for i,row in enumerate(conf['models']):
        key=row['name'];color=('#dc7633' if key==conf['primary_candidate'] else '#168578' if key==conf['matched_augmentation'] else '#3569a5' if key==conf['primary_baseline'] else '#8c959f')
        ax.plot(row['score'],i,'o',color=color,markersize=7)
        ax.annotate(f"{row['score']:.5f}",(row['score'],i),xytext=(7,0),textcoords='offset points',va='center',fontsize=9)
    ax.set_yticks(range(len(conf['models'])),[labels.get(r['name'],r['name']) for r in conf['models']])
    ax.invert_yaxis();ax.axvline(1.,ls='--',color='#566573',lw=1)
    ax.set_xlim(.960,1.108);ax.set_xlabel('Equal-year / horizon relative MSE (RW = 1; lower is better)')
    ax.set_title('A  Shanghai temporal confirmation, 2024–2025',loc='left',fontweight='bold',pad=14)
    ax.grid(axis='y',visible=False)
    comparisons=inference['blocks']['40']['comparisons']
    for index,key,title in [(0,conf['primary_baseline'],'B  Candidate minus selected price GRU'),(1,conf['matched_augmentation'],'C  Candidate minus matched augmentation')]:
        ax2=fig.add_subplot(gs[index,1]);c=comparisons[key]
        value=c['observed_delta'];lo,hi=c['basic_ci_95']
        ax2.errorbar(value,0,xerr=[[value-lo],[hi-value]],fmt='o',color='#dc7633',capsize=5,lw=2)
        ax2.axvline(0,color='#566573',ls='--',lw=1);ax2.set_yticks([])
        ax2.set_ylim(-1,1);ax2.set_title(title,loc='left',fontweight='bold',fontsize=10,pad=14)
        ax2.set_xlabel('Difference in relative MSE\n(negative favors candidate)',fontsize=9)
        ax2.text(.02,.90,f"Estimate {value:+.6f}\n95% CI [{lo:+.6f}, {hi:+.6f}]\nHolm-adjusted p = {c['holm_adjusted_p']:.3f}",transform=ax2.transAxes,va='top',fontsize=9)
        ax2.grid(axis='y',visible=False)
        ax2.ticklabel_format(axis='x',style='sci',scilimits=(-3,3),useMathText=True)
    fig.suptitle('The added conditional-moment mechanism did not pass confirmation',fontsize=14,fontweight='bold')
    fig.supxlabel('221 paired origins; 3 fixed training seeds; seed-mean losses. Registered moving blocks = 40 observations; 10,000 draws.\nPanels B and C have different horizontal scales. Intervals are conditional, approximate time-series intervals.',fontsize=9)
    for ext in ('png','svg'):fig.savefig(OUT/f'confirmation_results.{ext}')
    plt.close(fig)

    records=[]
    for stage in ('D1','D2'):
        records += json.loads((ROOT/(stage+'_RESULTS.json')).read_text())['results']
    ld=json.loads((ROOT/'LEARNING_DIAGNOSTIC_RESULTS.json').read_text())['results']
    model_ids={'Price GRU':'gru_price_h16_s0.1_l0','BCM-GRU':'kcm_h32_s1_l1','Matched augmentation':'augment_h32_s1_l0'}
    curve=[]
    for label,cid in model_ids.items():
        for selection in ('recent','spread'):
            for cap in (100,400,None):
                if cap is None:
                    selected=[r for r in records if r['task']['scope']=='pooled' and r['task']['config']['id']==cid
                              and r['task']['cutoff']=='2023-01-01' and r['task'].get('train_cap') is None]
                elif selection=='spread' or label=='Matched augmentation':
                    selected=[r for r in ld if r['task']['config']['id']==cid and r['task']['selection']==selection and r['task']['cap']==cap]
                else:
                    selected=[r for r in records if r['task']['scope']=='pooled' and r['task']['config']['id']==cid
                              and r['task']['cutoff']=='2023-01-01' and r['task'].get('train_cap')==cap]
                selected.sort(key=lambda r:r['task']['seed'])
                assert [r['task']['seed'] for r in selected]==[17,29,43]
                values=[r['metrics']['full']['score'] for r in selected]
                curve.append(dict(model=label,selection=selection,windows_per_market='all' if cap is None else cap,
                                  mean=float(np.mean(values)),seed_sd=float(np.std(values,ddof=1)),
                                  seed17=values[0],seed29=values[1],seed43=values[2]))
    with (ROOT/'results/LEARNING_CURVES.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(curve[0]),lineterminator='\n');w.writeheader();w.writerows(curve)
    fig,axes=plt.subplots(1,2,figsize=(11.4,5.0),sharey=True,layout='constrained')
    colors={'Price GRU':'#3569a5','BCM-GRU':'#dc7633','Matched augmentation':'#168578'}
    for ax,selection,title in zip(axes,['recent','spread'],['A  Most recent training origins','B  Uniformly spread across the same history']):
        for label in model_ids:
            rows=[r for r in curve if r['model']==label and r['selection']==selection]
            ax.errorbar(range(3),[r['mean'] for r in rows],yerr=[r['seed_sd'] for r in rows],label=label,
                        marker='o',capsize=3,color=colors[label],lw=1.8)
        ax.axhline(1.,ls='--',lw=1,color='#566573')
        ax.set_xticks(range(3),['100','400','All\n(1,033–1,171)'])
        ax.set_xlabel('Labelled windows per market');ax.set_title(title,loc='left',fontweight='bold',fontsize=11)
        ax.set_ylim(.93,1.185);ax.grid(axis='x',visible=False)
    axes[0].set_ylabel('2023 relative MSE (RW = 1; lower is better)')
    axes[1].legend(frameon=False,loc='upper left')
    fig.suptitle('Sample count and historical coverage have different effects',fontsize=14,fontweight='bold')
    fig.supxlabel('Four-market development evaluation; same full-history training scale and validation. Spread-100 is nested in spread-400.\nError bars show SD over 3 seeds, not independent time-series confidence intervals. No confirmation-driven model change.',fontsize=9)
    for ext in ('png','svg'):fig.savefig(OUT/f'learning_curves.{ext}')
    plt.close(fig)
    print(json.dumps(dict(figures=4,learning_curve_rows=len(curve),model_fits=0)))

if __name__=='__main__':main()
