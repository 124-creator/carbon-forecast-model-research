#!/usr/bin/env python3
"""Post-result diagnostic CIs for disclosed horizon and geography (NOT confirmatory)."""
import json, pathlib
import numpy as np
import pandas as pd
from run_experiment import run_cell, PRIVATE, RESULTS, SEED

def analysis():
  rng=np.random.default_rng(SEED+918)
  cellloss=[]
  for m in ['GUANGZHOU','HUBEI','SHANGHAI','EUA_LONG']:
    f=pd.read_csv(PRIVATE/(m+'_development.csv')).sort_values('date')
    date=pd.to_datetime(f.date).values.astype('datetime64[D]')
    lp=np.log(f.price.to_numpy(dtype=float))
    for y in [2022,2023]:
      _,loss=run_cell(m,y,lp,date,verbose=False)
      cellloss.append(loss)
  comparisons=['empirical_direct','selected_sqrt_mix','fixed_vr_mix']
  def get(n,hs,blk,reps=4000):
    traces={b:[] for b in comparisons}
    for _ in range(reps):
      metrics={m:[] for m in comparisons}
      for c in cellloss[:n]:
        N=len(c['selected_vr_mix'][5]);st=rng.integers(0,N,size=int(np.ceil(N/blk)))
        sel=np.concatenate([np.arange(s,s+blk)%N for s in st])[:N]
        z=np.mean([c['selected_vr_mix'][h][sel].mean() for h in hs]);
        for comp in comparisons:
          b=np.mean([c[comp][h][sel].mean() for h in hs]);
          metrics[comp].append((z,b))
      for comp in comparisons:
        z,b=np.asarray(metrics[comp]).mean(axis=0)
        traces[comp].append(100*(1-z/b))
    return {m:{'ci95':np.percentile(v,[2.5,97.5]).tolist(),'ci99':np.percentile(v,[.5,99.5]).tolist(),
               'p_lower_equal_zero':float(np.mean(np.array(v)<=0))} for m,v in traces.items()}
  out={'label':'POST_RESULT_DIAGNOSTIC_MULTIPLE_SPLITS_NOT_FROZEN_CONFIRMATION'}
  for n in [6,8]:
    for hs in [[5],[20],[5,20]]:
      for blk in [20,40]:
        k=f'{n}cells_H{"_".join(map(str,hs))}_block{blk}'
        out[k]=get(n,hs,blk)
  (RESULTS/'POSTHOC_BY_HORIZON_BOOTSTRAP.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
  print('Diagnostics complete: 12 horizon/market/length sets, 4000 paired bootstrap per comparison')
  for k in ['8cells_H5_20_block20','6cells_H20_block20','6cells_H5_block20']:
    print(k,out[k])
if __name__=='__main__':analysis()
