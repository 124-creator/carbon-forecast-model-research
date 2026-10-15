"""Fixed estimand and descriptive paired uncertainty; no model fitting."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

def score(pred,y,years):
 ratios=[];cells=[]
 for year in ['2020','2021']:
  mask=years==year
  for h in [1,5,20]:
   err=pred[:,mask,h-1]-y[mask,h-1]
   ratios.append(np.mean(err**2,axis=1)/np.mean(y[mask,h-1]**2))
   cells.append({'year':year,'h':h,'n':int(mask.sum()),'relative_MSE':float(np.mean(ratios[-1])),
                 'mean_seed_RMSE':float(np.mean(np.sqrt(np.mean(err**2,axis=1)))),'mean_seed_MAE':float(np.mean(np.abs(err))),
                 'ensemble_RMSE':float(np.sqrt(np.mean((pred.mean(0)[mask,h-1]-y[mask,h-1])**2)))})
 per_seed=np.mean(ratios,axis=0);ensemble=[]
 for year in ['2020','2021']:
  mask=years==year
  for h in [1,5,20]:ensemble.append(np.mean((pred.mean(0)[mask,h-1]-y[mask,h-1])**2)/np.mean(y[mask,h-1]**2))
 return {'score':float(per_seed.mean()),'ensemble':float(np.mean(ensemble)),'seeds':per_seed.tolist(),'cells':cells}

def run(out):
 out=Path(out)
 with np.load(out/'evaluation.npz',allow_pickle=False) as z:y=z['truth'];years=z['years']
 preds={}
 for fn in sorted((out/'predictions').glob('*.npz')):
  with np.load(fn,allow_pickle=False) as z:preds[fn.stem]=z['prediction']
 scores={k:score(v,y,years) for k,v in preds.items()};c=scores['CAUSAL']
 pool={k:v for k,v in scores.items() if k!='CAUSAL'};star=min(pool,key=lambda k:pool[k]['score']);estar=min(pool,key=lambda k:pool[k]['ensemble'])
 gains={k:1-c['score']/v['score'] for k,v in pool.items()}
 wins={k:int(np.sum(np.array(c['seeds'])<np.array(scores[k]['seeds']))) for k in ['IDENTITY','SYMMETRIC','SHUFFLED','FULL_WINDOW']}
 annual={year:float(np.mean([v['relative_MSE'] for v in c['cells'] if v['year']==year])) for year in ['2020','2021']}
 mechanism=all(gains[k]>=.01 for k in ['IDENTITY','SYMMETRIC','SHUFFLED']) and wins['IDENTITY']>=3 and gains['FROZEN']>0
 accuracy=gains['RW']>=.02 and gains[star]>=.01 and c['ensemble']<scores[estar]['ensemble'] and max(annual.values())<=1.05
 comparisons=list(dict.fromkeys(['RW','IDENTITY','SYMMETRIC','SHUFFLED','FULL_WINDOW',star]))
 boot={k:{} for k in comparisons}
 for length in [20,40,60]:
  values={k:np.zeros(2000) for k in ['CAUSAL']+comparisons};rng=np.random.default_rng(8400+length)
  for year in ['2020','2021']:
   mask=years==year;n=int(mask.sum());starts=rng.integers(0,n,(2000,int(np.ceil(n/length))));idx=((starts[:,:,None]+np.arange(length))%n).reshape(2000,-1)[:,:n]
   for h in [1,5,20]:
    rw=(y[mask,h-1]**2)[idx].mean(1)
    for name in values:
     loss=np.mean((preds[name][:,mask,h-1]-y[mask,h-1])**2,axis=0)
     values[name]+=loss[idx].mean(1)/rw/6
  for name in comparisons:boot[name][str(length)]=np.quantile(100*(1-values['CAUSAL']/values[name]),[.025,.975]).tolist()
 result={'status':'DEVELOPMENT_GATE_PASSED_NOT_NOVELTY_CERTIFIED' if mechanism and accuracy else 'STOP_DEVELOPMENT_GATE_FAILED',
         'candidate':c,'scores':scores,'Cstar':star,'Cstar_ensemble':estar,'gains':gains,'paired_seed_wins':wins,'annual':annual,
         'mechanism_pass':bool(mechanism),'accuracy_pass':bool(accuracy),'bootstrap_percent_95':boot,'configuration_count':len(preds),
         'scope':'repeatedly exposed development; no selection adjustment, no holdout or publication certification','later_period_read':False,'novelty_established':False}
 (out/'RESULTS.json').write_text(json.dumps(result,indent=2))
 pd.DataFrame([{'name':k,'mean_seed_relative_MSE':v['score'],'prediction_ensemble_relative_MSE':v['ensemble']} for k,v in scores.items()]).to_csv(out/'all_scores.csv',index=False)
 print(json.dumps({k:result[k] for k in ['status','Cstar','gains','paired_seed_wins','mechanism_pass','accuracy_pass']},indent=2));return result
