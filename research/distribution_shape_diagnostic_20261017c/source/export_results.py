"""Export public aggregates and scientific figures; no quote/weight/NPZ access."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def write_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--result',required=True);ap.add_argument('--study',required=True)
    args=ap.parse_args();result=json.loads(Path(args.result).read_text());study=Path(args.study)
    if result['status']!='complete':raise ValueError('No complete result to chart')
    table=study/'results/tables';fig=study/'figures';table.mkdir(parents=True,exist_ok=True);fig.mkdir(exist_ok=True)
    names={'fhs_fixed':'Fixed empirical shape','fhs_rolling':'Rolling empirical shape','ewma_normal':'EWMA + normal','ewma_t5':'EWMA + t5',
           'garch_normal':'GARCH normal','garch_t5':'GARCH t5','garch_fhs':'GARCH-FHS (31 atoms)',
           'global_shape':'Global learned shape','conditional_shape':'Conditional shape','scale_only':'Scale only',
           'shape_scale':'Shape + scale','deepnpts_rps':'DeepNPTS core / author score','deepnpts_crps':'DeepNPTS core / exact CRPS'}
    model_rows=[dict(model=k,score=v,reference='same-cell fixed empirical shape',primary_horizons='1,5,20') for k,v in sorted(result['scores'].items(),key=lambda z:z[1])]
    write_csv(table/'models.csv',model_rows)
    market_rows=[dict(model=f,market=m,score=v) for f,markets in result['market_scores'].items() for m,v in markets.items()]
    write_csv(table/'markets.csv',market_rows)
    lookup={(r['market'],r['family']):r for r in result['rows']}
    horizons=[];ledger=[]
    for r in result['rows']:
        for h in range(20):
            q=r['metrics'];den=lookup[r['market'],'fhs_fixed']['metrics']['crps'][h]
            horizons.append(dict(model=r['family'],market=r['market'],horizon=h+1,n=q['n'],crps=q['crps'][h],
                relative_crps=q['crps'][h]/den,mean_mse_ratio=q['mean_mse_ratio'][h],
                coverage50=q['coverage_50'][h],coverage80=q['coverage_80'][h],coverage90=q['coverage_90'][h],width90=q['width_90'][h]))
        ledger.append(dict(model=r['family'],market=r['market'],fitted=r.get('fitted',False),seconds=r['seconds'],
            epochs=r.get('completed_epochs',0),steps=r.get('optimizer_steps',0),selected_epoch=r.get('selected_epoch'),
            validation_score=r.get('validation_score'),status=r['status']))
    write_csv(table/'horizons.csv',horizons);write_csv(table/'execution.csv',ledger)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    f,ax=plt.subplots(figsize=(10.5,6.7),constrained_layout=True)
    labels=[r['model'] for r in model_rows];values=[r['score'] for r in model_rows]
    colors=['#237a72' if s in ('conditional_shape','shape_scale') else '#46699c' if s in ('global_shape','scale_only') else '#9aa5b1' for s in labels]
    y=np.arange(len(labels));ax.barh(y,values,color=colors);ax.set_yticks(y,[names[s] for s in labels]);ax.invert_yaxis()
    ax.axvline(1,color='#333333',linestyle='--',linewidth=1)
    ax.set_xlim(0,max(values)*1.12);ax.set_xlabel('Mean CRPS ratio to fixed empirical shape (lower is better)')
    ax.set_title('2022 development diagnostic | 5 series, horizons 1 / 5 / 20\n13 finite-distribution forecasts; these are not novel algorithms',loc='left',fontsize=12,pad=12)
    for yy,v in zip(y,values):ax.text(v+.004,yy,f'{v:.4f}',va='center',fontsize=9)
    for ext in ('png','svg'):f.savefig(fig/f'D1_all_models.{ext}',dpi=200)
    plt.close(f)
    markets=list(result['market_scores']['conditional_shape'])
    gains1=[100*(1-result['market_scores']['conditional_shape'][m]/result['market_scores']['global_shape'][m]) for m in markets]
    gains2=[100*(1-result['market_scores']['shape_scale'][m]/result['market_scores']['scale_only'][m]) for m in markets]
    f,ax=plt.subplots(figsize=(10.5,4.6),constrained_layout=True);x=np.arange(len(markets))
    ax.bar(x-.18,gains1,.36,label='Conditional shape vs global shape',color='#237a72')
    ax.bar(x+.18,gains2,.36,label='Shape + scale vs scale only',color='#46699c')
    ax.axhline(0,color='#333333',linewidth=.8);ax.set_xticks(x,[m.replace('EEX_EUA_T3PA','EEX EUA').replace('EU_REPORTED','EU reported') for m in markets],fontsize=9)
    ax.set_ylabel('Relative score reduction (%) | positive is improvement')
    ax.set_title('Matched comparisons: does shape add useful information?\nEach bar uses the mean of three same-market normalized CRPS cells',loc='left',fontsize=12,pad=12)
    ax.legend(frameon=False,loc='best')
    for ext in ('png','svg'):f.savefig(fig/f'D1_matched_comparisons.{ext}',dpi=200)
    plt.close(f)
    receipt={'source_result_sha256':hashlib.sha256(Path(args.result).read_bytes()).hexdigest(),'market_quote_reads':0,'fits':0,
             'table_rows':{'models':len(model_rows),'markets':len(market_rows),'horizons':len(horizons),'execution':len(ledger)},
             'plot_note':'Development means, no confidence intervals or significance claim. All hypotheses and gates remain as frozen.'}
    (study/'provenance/TABLE_FIGURE_EXPORT.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
