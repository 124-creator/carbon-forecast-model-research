"""Summaries and paired dependent-data uncertainty; no extra training/selection."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from lowdim import targets,H
from experiment import load_market_prices

def main(root,out):
    result=json.loads((out/'market_tuning.json').read_text())
    archive=np.load(out/'market_tuning_predictions.npz',allow_pickle=False)
    d=load_market_prices(root);dates=pd.DatetimeIndex(pd.to_datetime(d.date));p=d.author_reported_price.to_numpy(float)
    selected=result['candidate'];cstar=result['Cstar'];groups=[]
    for year in [2020,2021]:
        ix=np.flatnonzero(dates.year==year);o=ix[ix+20<=ix[-1]];y=targets(p,o)
        groups.append(np.stack([y*y,(archive[selected][o]-y)**2,(archive[cstar][o]-y)**2],axis=1))
    boot=[]
    for block in [20,40,60]:
        rng=np.random.default_rng(20261013);draw=[]
        for _ in range(2000):
            scores=[]
            for g in groups:
                n=len(g);starts=rng.integers(0,n-block+1,size=int(np.ceil(n/block)))
                idx=(starts[:,None]+np.arange(block)[None,:]).ravel()[:n]
                losses=g[idx].mean(axis=0);scores.append((losses/losses[0]).mean(axis=1))
            s=np.mean(scores,axis=0);draw.append([100*(1-s[1]/s[0]),100*(1-s[1]/s[2])])
        q=np.quantile(draw,[.025,.975],axis=0)
        boot.append({'block':block,'method':'paired noncircular moving-block bootstrap independently within each year','repetitions':2000,'gain_vs_RW_95_percent':q[:,0].tolist(),'gain_vs_Cstar_95_percent':q[:,1].tolist(),'scope':'tuning-selected estimates; no model-selection correction, no confirmation claim'})
    table=pd.read_csv(out/'market_tuning_all.csv');best=table.loc[table.groupby('family').score.idxmin()].sort_values('score')
    summary=dict(best_by_family=best.to_dict('records'),candidate=selected,Cstar=cstar,gains=result['gains'],tuning_gate_passed=result['tuning_gate_passed'],max_abs_candidate_Cstar_difference=float(np.nanmax(abs(archive[selected]-archive[cstar]))),bootstrap=boot)
    (out/'summary.json').write_text(json.dumps(summary,indent=2));best.to_csv(out/'best_by_family.csv',index=False)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('.'));ap.add_argument('--out',type=Path,default=Path('outputs'));a=ap.parse_args();main(a.root,a.out)
